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
//! See `docs/explanation/async-design.md` and `docs/how-to/run-under-async.md` for the full design
//! and gotchas (worker pool is opt-in, defaults to one background worker).
//!
//! `func`/`data1` may now run on a different OS thread than the caller (the
//! scheduler's worker pool), where `rb_thread_call_without_gvl` always ran
//! them on the calling thread itself. `without_gvl` therefore requires
//! `F: Send` (the closure crosses onto the worker thread) and `R: Send` (its
//! result crosses back) — the compiler enforces the cross-thread contract
//! instead of leaving it to call-site audit. Every current `without_gvl`
//! call site (`tokenizer.rs`'s `encode_batch`/`encode_files`) already shares
//! its `&Tokenizer`/`&WorkerPool` across `rayon` worker threads inside
//! `encode_docs_ragged`. The BPE batch paths' input marshal (`tokenizer.rs`'s
//! `marshal_inputs`) borrows heap `RString` buffers zero-copy where it can
//! (locked via `rb_str_locktmp`, or unlocked when the string is frozen) and
//! falls back to an owned `Vec<u8>` copy otherwise — either way, only raw
//! byte slices are ever captured by the closure passed here, never a Ruby
//! `VALUE` or thread-local state, so running one OS thread over instead of
//! another changes nothing about its safety, and the types involved satisfy
//! `Send`/`Sync` on their own merits.
//!
//! Interrupts. `rb_nogvl` ends by checking interrupts and raises any that
//! are pending — a `Timeout`, `Thread#kill`, `Interrupt`, an `Async`
//! timeout — by longjmping out of itself, over every Rust frame in between.
//! Both entry points here therefore call it under `magnus::rb_sys::protect`
//! (see [`run`]), so the raise comes back as an ordinary `Error` and every
//! guard, `InputDocs` and Ruby String lock the caller holds is released by
//! ordinary Rust unwinding on the way out. [`without_gvl_cancellable`] goes
//! one further and supplies a real unblock function, so a pending interrupt
//! cancels the work in flight instead of waiting for it.

use std::any::Any;
use std::ffi::c_void;
use std::os::raw::c_int;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};

use magnus::Error;
use rb_sys::rb_nogvl;

/// `RB_NOGVL_OFFLOAD_SAFE` (`ruby/thread.h:73` in Ruby 4.0.7's own headers;
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
/// catching it here and resuming it from [`Slot::take`] below turns that
/// into an ordinary Rust panic, which magnus's own `method!`/`function!`
/// call trampolines already wrap in `catch_unwind` and convert into a fatal
/// Ruby exception (`magnus::error::Error::from_panic`) — the same outcome
/// any other panicking native method already gets, just carried safely
/// across the extra C boundary this one call adds.
enum Outcome<R> {
    Value(R),
    Panic(Box<dyn Any + Send + 'static>),
}

/// The one thing `rb_nogvl`'s callback and [`run`] share: the callback takes
/// the closure out of `input` and leaves its result in `output`. The rule is
/// that nothing travels through `rb_nogvl`'s *return value* — the caller owns
/// this slot, in an ordinary Rust frame.
///
/// That rule is what makes the raise `rb_nogvl` performs on its way out
/// leak-free. The `protect` closure in [`run`] holds the single FFI call and
/// owns nothing at all, so the longjmp skips no destructor that matters:
/// whatever is in flight is either here — in a frame that unwinds normally
/// afterwards — or in one of the caller's own frames, which unwind with it.
/// It is also the only way to tell "the callback ran" from "it never did",
/// which a fiber scheduler cancelling the offloaded operation before its
/// worker pool picks it up makes a real case (io-event's
/// `worker_pool_work_wait`, `ext/io/event/worker_pool.c`).
struct Slot<F, R> {
    input: Option<F>,
    output: Option<Outcome<R>>,
}

impl<F, R> Slot<F, R> {
    fn new(f: F) -> Self {
        Self {
            input: Some(f),
            output: None,
        }
    }

    /// The callback's value, or `None` if it never ran. A panic it caught
    /// resumes here, on an ordinary Rust frame.
    fn take(&mut self) -> Option<R> {
        match self.output.take() {
            Some(Outcome::Value(value)) => Some(value),
            Some(Outcome::Panic(payload)) => panic::resume_unwind(payload),
            None => None,
        }
    }
}

unsafe extern "C" fn call_without_gvl<F, R>(arg: *mut c_void) -> *mut c_void
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    // SAFETY: `arg` is the `*mut Slot<F, R>` handed to `rb_nogvl` by `run`,
    // valid for the duration of that (synchronous) call, and this is the
    // only place it's dereferenced.
    let slot = unsafe { &mut *(arg as *mut Slot<F, R>) };
    let closure = slot
        .input
        .take()
        .expect("without_gvl callback invoked more than once");
    slot.output = Some(match panic::catch_unwind(AssertUnwindSafe(closure)) {
        Ok(value) => Outcome::Value(value),
        Err(payload) => Outcome::Panic(payload),
    });
    std::ptr::null_mut()
}

/// Run `slot`'s closure with the GVL released, under `rb_protect`.
///
/// `rb_nogvl` finishes by checking interrupts and raising any that are
/// pending, by longjmping out of itself. `protect` catches that and returns
/// it as an `Error` — including the `Tag::Fatal` a `Thread#kill` jumps with,
/// which magnus resumes with `rb_jump_tag` once the caller's frames have
/// unwound. The closure below is deliberately trivial: one FFI call and a
/// `nil`, owning nothing the longjmp could strand.
fn run<F, R>(
    slot: &mut Slot<F, R>,
    ubf: rb_sys::rb_unblock_function_t,
    data2: *mut c_void,
) -> Result<(), Error>
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    let arg = slot as *mut Slot<F, R> as *mut c_void;
    let nil: rb_sys::VALUE = rb_sys::Qnil.into();
    magnus::rb_sys::protect(|| {
        // SAFETY: `arg` points at `slot`, which outlives this synchronous
        // call; the callback is the only reader of it (see `Slot`).
        unsafe {
            rb_nogvl(
                Some(call_without_gvl::<F, R>),
                arg,
                ubf,
                data2,
                RB_NOGVL_OFFLOAD_SAFE,
            )
        };
        nil
    })?;
    Ok(())
}

/// `rb_nogvl`'s unblock function: Ruby calls it from another thread when an
/// interrupt is pending for this one, and a fiber scheduler calls it to
/// cancel an offloaded operation (`rb_fiber_scheduler_blocking_operation_cancel`
/// "marks it as cancelled and calls the unblock function", Ruby 4.0.7's
/// `ruby/fiber/scheduler.h:455-457` — which is how an `Async` timeout reaches
/// an encode running on io-event's worker pool). It does exactly one thing:
/// set the flag the core's encode loops poll (`src/batch.rs`), so the encode
/// stops at the next document boundary instead of running to completion.
unsafe extern "C" fn set_cancel(arg: *mut c_void) {
    // SAFETY: `arg` is the `&AtomicBool` `without_gvl_cancellable` passed as
    // `data2`, living in its frame for the whole `rb_nogvl` call — the only
    // window in which Ruby may call this.
    unsafe { &*(arg as *const AtomicBool) }.store(true, Ordering::Relaxed);
}

/// Run `f` with the GVL released: other Ruby threads may run while `f`
/// executes, and — under a fiber scheduler with a worker pool — the calling
/// fiber yields to the reactor while `f` runs on a background thread. `f`
/// must not touch any Ruby object (`VALUE`) — only plain Rust data — per the
/// Ruby C API's contract for this call. An interrupt that arrives meanwhile
/// is delivered once `f` has finished, as an `Err`; use
/// [`without_gvl_cancellable`] for work that can stop early.
pub fn without_gvl<F, R>(f: F) -> Result<R, Error>
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    let mut slot = Slot::new(f);
    run(&mut slot, None, std::ptr::null_mut())?;
    Ok(match slot.take() {
        Some(value) => value,
        // A fiber scheduler cancelled the offloaded operation before its
        // worker pool ever started it, and nothing was raised (that would
        // have come back as an `Err` above). The closure is still in the
        // slot and still ours to run: do that here, GVL and all, rather
        // than invent a result.
        None => slot.input.take().expect("the callback left the closure")(),
    })
}

/// [`without_gvl`] for work that can stop early: `attempt` builds the
/// closure, which gets a cancellation token to poll and returns `None` if it
/// saw the token set and cut its work short. An interrupt arriving during
/// the run sets that token through [`set_cancel`], so the caller sees the
/// interrupt instead of waiting out the whole batch.
///
/// `attempt` is a factory, not the closure itself, because a cancelled run
/// sometimes has to be redone: Ruby calls the unblock function for every
/// interrupt, including ones that never raise (a trap handler, a
/// `Thread#wakeup`), and returning the half-encoded batch those produce
/// would be a silently truncated result. The redo runs uninterruptibly —
/// plain [`without_gvl`], which is exactly how this path behaved before
/// cancellation existed — so a chatty signal handler can cost a batch one
/// extra pass, never an unbounded number of them.
pub fn without_gvl_cancellable<A, F, R>(mut attempt: A) -> Result<R, Error>
where
    A: FnMut() -> F,
    F: FnOnce(&AtomicBool) -> Option<R> + Send,
    R: Send,
{
    let cancel = AtomicBool::new(false);
    let f = attempt();
    let mut slot = Slot::new(|| f(&cancel));
    run(
        &mut slot,
        Some(set_cancel),
        &cancel as *const AtomicBool as *mut c_void,
    )?;
    if let Some(Some(value)) = slot.take() {
        return Ok(value);
    }
    let f = attempt();
    let never_cancelled = AtomicBool::new(false);
    without_gvl(|| f(&never_cancelled))
        .map(|value| value.expect("nothing sets the token of an uncancellable run"))
}
