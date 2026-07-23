//! Release Ruby's GVL (global VM lock) for the duration of a CPU-bound Rust
//! closure, so other Ruby threads/fibers keep running while the core worker
//! pool chews through a batch encode.
//!
//! Magnus 0.8 does not wrap `rb_thread_call_without_gvl` — magnus's own API
//! coverage notes (magnus-0.8.2/src/lib.rs, the `## rb_t` section) list it as
//! unimplemented, alongside `rb_nogvl`. The `rb-sys` crate (already an
//! indirect dependency via magnus) carries the raw bindgen-generated symbol,
//! but reaching it would mean adding `rb-sys` as a direct dependency of this
//! crate, which perturbs the workspace `Cargo.lock` outside this lane's
//! `ext/**`/`lib/**`/`spec/**` boundary — see the builder report's
//! DISAGREEMENTS section. Instead this declares the same long-stable
//! `ruby/thread.h` C signature by hand: the symbol is resolved at load time
//! against the embedding Ruby process (the same `dynamic_lookup` mechanism
//! every `rb_*` call in this extension already relies on), so no extra
//! dependency — and no `Cargo.lock` edit — is needed.

use std::ffi::c_void;

unsafe extern "C" {
    fn rb_thread_call_without_gvl(
        func: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
        data1: *mut c_void,
        ubf: Option<unsafe extern "C" fn(*mut c_void)>,
        data2: *mut c_void,
    ) -> *mut c_void;
}

unsafe extern "C" fn call_without_gvl<F, R>(arg: *mut c_void) -> *mut c_void
where
    F: FnOnce() -> R,
{
    // SAFETY: `arg` is the `*mut Option<F>` handed to
    // `rb_thread_call_without_gvl` below, valid for the duration of that
    // (synchronous) call, and this is the only place it's dereferenced.
    let closure = unsafe { (*(arg as *mut Option<F>)).take() }
        .expect("without_gvl callback invoked more than once");
    Box::into_raw(Box::new(closure())) as *mut c_void
}

/// Run `f` with the GVL released: other Ruby threads may run while `f`
/// executes on this OS thread. `f` must not touch any Ruby object (`VALUE`)
/// — only plain Rust data — per the Ruby C API's contract for this call.
pub fn without_gvl<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let mut slot = Some(f);
    let arg = &mut slot as *mut Option<F> as *mut c_void;
    let result = unsafe {
        rb_thread_call_without_gvl(call_without_gvl::<F, R>, arg, None, std::ptr::null_mut())
    };
    // SAFETY: `result` is the `Box::into_raw(Box::new(R))` pointer produced
    // by the callback above, which always runs exactly once before
    // `rb_thread_call_without_gvl` returns.
    *unsafe { Box::from_raw(result as *mut R) }
}
