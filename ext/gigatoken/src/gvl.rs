//! Release Ruby's GVL (global VM lock) for the duration of a CPU-bound Rust
//! closure, so other Ruby threads/fibers keep running while the core worker
//! pool chews through a batch encode.
//!
//! Uses `rb_sys::rb_nogvl` (this crate's direct `rb-sys` dependency, the same
//! version magnus already resolves — see `ext/gigatoken/Cargo.toml`) with the
//! `RB_NOGVL_OFFLOAD_SAFE` flag, rather than plain `rb_thread_call_without_gvl`.
//! Under a `Fiber.scheduler` built with a worker pool (`Async::Scheduler.new`
//! with `ASYNC_SCHEDULER_WORKER_POOL=true` or an explicit `worker_pool:`,
//! `~/architect/src/github.com/socketry/async/lib/async/scheduler.rb:31-101`),
//! `OFFLOAD_SAFE` tells Ruby it's safe to hand `func`/`data1` to the
//! scheduler's `blocking_operation_wait` hook instead of just blocking this OS
//! thread: the scheduler runs the closure on an `IO::Event::WorkerPool`
//! background thread while transferring the calling *fiber* (not just other
//! Threads) back to the reactor, so other fibers on this thread keep running
//! for the encode's duration
//! (`~/architect/src/github.com/socketry/io-event/ext/io/event/worker_pool.c:309-316`).
//! Without such a scheduler (or with one lacking a worker pool), `rb_nogvl`
//! degrades to exactly today's behavior: release the GVL, block this thread.
//! See `docs/rb/async-design.md` and `docs/rb/async.md` for the full design
//! and gotchas (worker pool is opt-in, defaults to one background worker).
//!
//! `func`/`data1` may now run on a different OS thread than the caller (the
//! scheduler's worker pool), where `rb_thread_call_without_gvl` always ran
//! them on the calling thread itself. Every current `without_gvl` call site
//! (`tokenizer.rs`'s `encode_batch`/`encode_files`) already shares its
//! `&Tokenizer`/`&WorkerPool` across `rayon` worker threads inside
//! `encode_docs_ragged`, and copies every input `RString` into an owned
//! `Vec<u8>` before calling `without_gvl` at all — nothing captured touches a
//! Ruby `VALUE` or thread-local state, so running one OS thread over instead
//! of another changes nothing about its safety.

use std::any::Any;
use std::ffi::c_void;
use std::os::raw::c_int;
use std::panic::{self, AssertUnwindSafe};

use rb_sys::rb_nogvl;

/// `RB_NOGVL_OFFLOAD_SAFE` (`ruby/thread.h:84` in a current Ruby checkout;
/// introduced by Ruby's `Fiber::Scheduler#blocking_operation_wait` support,
/// first released in Ruby 3.4.0). Defined locally rather than taken from
/// `rb_sys::` bindings: `rb-sys` bindgens its constants from the *building*
/// Ruby's own headers, and this gem's floor (`gigatoken.gemspec`, `>=
/// 3.3.0`) may build against 3.3 headers that don't declare this macro at
/// all. Verified against Ruby 3.3.0's `thread.c` (`rb_nogvl`, ~line 1508):
/// it only tests `RB_NOGVL_UBF_ASYNC_SAFE`/`RB_NOGVL_INTR_FAIL` against
/// `flags`, so an unrecognized bit here is silently ignored — passing it on
/// 3.3 degrades to plain blocking `rb_nogvl`, identical to the old
/// `rb_thread_call_without_gvl`. No `RUBY_VERSION` gate is needed.
const RB_NOGVL_OFFLOAD_SAFE: c_int = 0x4;

/// The result of running `f` inside `call_without_gvl`: either its return
/// value, or a caught panic payload to re-raise once we're back on ordinary
/// (non-`extern "C"`) Rust stack frames. Unwinding a panic directly across
/// the `extern "C"` trampoline `rb_nogvl` calls into is undefined behavior;
/// catching it here and resuming it from `without_gvl` below turns that into
/// an ordinary Rust panic, which magnus's own `method!`/`function!` call
/// trampolines already wrap in `catch_unwind` and convert into a fatal Ruby
/// exception (`magnus::error::Error::from_panic`) — the same outcome any
/// other panicking native method already gets, just carried safely across
/// the extra C boundary this one call adds.
enum Outcome<R> {
    Value(R),
    Panic(Box<dyn Any + Send + 'static>),
}

unsafe extern "C" fn call_without_gvl<F, R>(arg: *mut c_void) -> *mut c_void
where
    F: FnOnce() -> R,
{
    // SAFETY: `arg` is the `*mut Option<F>` handed to `rb_nogvl` below, valid
    // for the duration of that (synchronous) call, and this is the only
    // place it's dereferenced.
    let closure = unsafe { (*(arg as *mut Option<F>)).take() }
        .expect("without_gvl callback invoked more than once");
    let outcome = match panic::catch_unwind(AssertUnwindSafe(closure)) {
        Ok(value) => Outcome::Value(value),
        Err(payload) => Outcome::Panic(payload),
    };
    Box::into_raw(Box::new(outcome)) as *mut c_void
}

/// Run `f` with the GVL released: other Ruby threads may run while `f`
/// executes, and — under a fiber scheduler with a worker pool — the calling
/// fiber yields to the reactor while `f` runs on a background thread. `f`
/// must not touch any Ruby object (`VALUE`) — only plain Rust data — per the
/// Ruby C API's contract for this call. A panic inside `f` is caught and
/// re-raised here rather than left to unwind across the C trampoline.
pub fn without_gvl<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let mut slot = Some(f);
    let arg = &mut slot as *mut Option<F> as *mut c_void;
    let result = unsafe {
        rb_nogvl(
            Some(call_without_gvl::<F, R>),
            arg,
            None,
            std::ptr::null_mut(),
            RB_NOGVL_OFFLOAD_SAFE,
        )
    };
    // SAFETY: `result` is the `Box::into_raw(Box::new(Outcome<R>))` pointer
    // produced by the callback above, which always runs exactly once before
    // `rb_nogvl` returns.
    let outcome = *unsafe { Box::from_raw(result as *mut Outcome<R>) };
    match outcome {
        Outcome::Value(value) => value,
        Outcome::Panic(payload) => panic::resume_unwind(payload),
    }
}
