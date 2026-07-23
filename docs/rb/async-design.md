# Async-friendly batch/file encodes (BRIEF §5 spike)

## Question

`BPETokenizer#encode_batch` / `#encode_files` (`ext/gigatoken/src/tokenizer.rs:72-118`)
already release the GVL for the encode via `gvl::without_gvl`
(`ext/gigatoken/src/gvl.rs`), which wraps `rb_thread_call_without_gvl`. That
lets other **Ruby Threads** run during a long encode, but it does nothing for
**fibers**: `rb_thread_call_without_gvl` blocks the calling OS thread until
the closure returns, and under `Async::Scheduler` the reactor and every other
fiber live on that same thread. A long `encode_files` call today stalls the
whole reactor for its duration.

The question: how do we hand the encode to the Rust worker pool and let the
calling **fiber** (not just other threads) yield to the reactor until it's
done — without spawning Ruby threads ourselves?

## Recommendation: ADOPT `rb_nogvl(..., RB_NOGVL_OFFLOAD_SAFE)`

Ruby ships a purpose-built mechanism for exactly this, one level below the
FD/pipe design sketched in the iteration brief. **No new Ruby-visible API is
needed.** The fix is entirely inside `ext/gigatoken/src/gvl.rs`: replace the
hand-declared `rb_thread_call_without_gvl` extern with the newer `rb_nogvl`,
passing the `RB_NOGVL_OFFLOAD_SAFE` flag.

```
ruby/thread.h:73   #define RB_NOGVL_OFFLOAD_SAFE  (0x4)
ruby/thread.h:187  void *rb_nogvl(void *(*func)(void *), void *data1,
                                  rb_unblock_function_t *ubf, void *data2,
                                  int flags);
```

(from
`/Users/eric/.local/share/mise/installs/ruby/4.0.6/include/ruby-4.0.0/ruby/thread.h`,
lines 62-73 and 187-189 — this Ruby is the ext's actual build/runtime target
in this worktree.)

Per the doc comment at `thread.h:62-71`: this flag "indicates that the passed
function is safe to run using a fiber scheduler's `blocking_operation_wait`
hook." Concretely:

- If the current `Fiber.scheduler` implements `blocking_operation_wait`
  (`Async::Scheduler` does, see below), Ruby hands `func`/`data1` to the
  scheduler instead of just releasing the GVL in place. `Async::Scheduler`
  forwards it to an `IO::Event::WorkerPool`
  (`~/architect/src/github.com/socketry/async/lib/async/scheduler.rb:52-63`),
  which runs it on one of *its own* background Ruby Threads (created via
  `rb_thread_create` inside `io-event`'s C extension — not our code) while
  **transferring the calling fiber back to the reactor**
  (`~/architect/src/github.com/socketry/io-event/ext/io/event/worker_pool.c:309-316`,
  `rb_fiber_scheduler_block`), so other fibers on the same thread keep
  running. When the work finishes, the worker thread calls
  `rb_fiber_scheduler_unblock` (`worker_pool.c:217`), which pushes the
  waiting fiber back onto the reactor's ready list.
- If no scheduler is active, or the active scheduler doesn't implement the
  hook, `rb_nogvl` degrades to today's behavior: it just releases the GVL and
  blocks the calling thread, same as `rb_thread_call_without_gvl`. Nothing
  breaks for callers outside `Async { }`.

This is a **strict superset** of the current `without_gvl`, with identical
call/return semantics (synchronous, same thread resumes with the GVL held) —
only whether *other fibers on this thread* can run meanwhile changes, and
only when the app has opted in (see Gotcha below). No `IO` object, no pipe,
no eventfd, no new Ruby method.

### API sketch

No new method. The existing synchronous API is unchanged:

```ruby
Async do
  tokenizer.encode_files(source)   # yields the calling fiber while the
end                                # Rust encode runs; other fibers/tasks
                                    # in this Async block keep running.
```

Rust side (`ext/gigatoken/src/gvl.rs`), the only change from what's there
today:

```rust
unsafe extern "C" {
    fn rb_nogvl(
        func: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
        data1: *mut c_void,
        ubf: Option<unsafe extern "C" fn(*mut c_void)>,
        data2: *mut c_void,
        flags: c_int,
    ) -> *mut c_void;
}
const RB_NOGVL_OFFLOAD_SAFE: c_int = 0x4;

pub fn without_gvl<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    // same Box<Option<F>>/call_without_gvl plumbing as today, just:
    let result = unsafe {
        rb_nogvl(call_without_gvl::<F, R>, arg, None, std::ptr::null_mut(), RB_NOGVL_OFFLOAD_SAFE)
    };
    ...
}
```

`tokenizer.rs`'s `encode_batch`/`encode_files` bodies do not change at all —
they already call `without_gvl(...)`.

### Gotcha callers must know: worker pool is opt-in, not automatic

`Async::Scheduler#blocking_operation_wait` only exists if the scheduler was
built with a worker pool
(`~/architect/src/github.com/socketry/async/lib/async/scheduler.rb:31-33,78-101`):

```ruby
WORKER_POOL = ENV.fetch("ASYNC_SCHEDULER_WORKER_POOL", nil).then do |value|
  value == "true" ? true : nil
end
...
def initialize(parent = nil, selector: nil, profiler: Profiler&.default, worker_pool: WORKER_POOL)
  ...
  if @worker_pool
    self.singleton_class.prepend(BlockingOperationWait)
  end
end
```

A bare top-level `Async { ... }` (`Kernel#Async`,
`~/architect/src/github.com/socketry/async/lib/kernel/async.rb`, and
`Async::Reactor#initialize`,
`~/architect/src/github.com/socketry/async/lib/async/reactor.rb:21-25`) does
**not** set `worker_pool: true` by default. Without it, `rb_nogvl(...,
RB_NOGVL_OFFLOAD_SAFE)` finds no `blocking_operation_wait` hook and falls
back to blocking the whole reactor thread — exactly today's behavior, silently.

So: an app that wants `encode_files` to actually cooperate with other fibers
must run with `ASYNC_SCHEDULER_WORKER_POOL=true` in the environment (or
construct its own `Async::Scheduler.new(worker_pool: IO::Event::WorkerPool.new(...))`
and `Fiber.set_scheduler` it). This is a documented, one-line operational
requirement, not a code change — but it belongs in gigatoken's README /
CHANGELOG the moment this ships, or callers will quietly get the old
blocking behavior. Async's own release notes
(`~/architect/src/github.com/socketry/async/releases.md:378-390`) also flag
that the worker pool has real overhead and should be benchmarked, not
enabled blindly:

> It should be noted that this isn't a net win, as the overhead of using a
> worker pool can be significant compared to the `rb_nogvl` work.

The default `IO::Event::WorkerPool` also caps at **one** background worker
thread (`worker_pool.c:252-273`, `maximum_worker_count = 1` default) — one
in-flight offloaded encode at a time; a second concurrent
`encode_batch`/`encode_files` call from another fiber queues behind it. Apps
running several concurrent encodes would want a bigger pool
(`Async::Scheduler.new(worker_pool: IO::Event::WorkerPool.new(maximum_worker_count: N))`),
which is an app-level tuning knob, not something gigatoken-rb should hardcode.

## Safety argument

**GVL.** Unchanged from today's `without_gvl`: the closure passed to
`rb_nogvl` must not touch any `VALUE` — plain Rust data only — exactly the
existing contract documented at `ext/gigatoken/src/gvl.rs:41-43`. `rb_nogvl`
is `rb_thread_call_without_gvl`'s direct successor (same header, adjacent
declarations, `thread.h:106-189`) and Ruby's own docs give it the identical
"no Ruby C API calls inside `func`" warning (`thread.h:130-133`).

**GC / object lifetime.** No change needed: `encode_batch` and
`encode_files` (`ext/gigatoken/src/tokenizer.rs:72-118`) already copy every
input `RString` into an owned `Vec<u8>` *before* calling `without_gvl`, per
the existing doc comment at `tokenizer.rs:68-71` ("Every input string is
copied into an owned buffer before release: nothing Ruby-managed may be
touched once the GVL is gone"). The offloaded closure only ever sees owned
buffers and the `WorkerPool`/`Tokenizer` handles already `RefCell`-borrowed
before the call — identical whether it runs on the calling thread (today) or
on an `io-event` worker thread (this recommendation). Nothing new is pinned
or exposed to the collector; `io-event`'s `WorkerPool` itself has its own GC
write-barrier/compaction handling for the objects *it* tracks
(`~/architect/src/github.com/socketry/io-event/releases.md:48`, "Improve
`WorkerPool` GC compaction support... fixing potential use-after-free under
compacting GC") — that's `io-event`'s problem, not ours, since we hand it no
`VALUE`s.

**Panic boundary — pre-existing gap, not introduced by this change.**
`gvl.rs`'s current `call_without_gvl<F, R>` (`ext/gigatoken/src/gvl.rs:29-39`)
calls `closure()` with no `catch_unwind`. If the Rust encode panics —
plausible, since `WorkerPool::with_worker`
(`src/batch.rs:697-726`) already anticipates and recovers from a worker
**mutex being poisoned by a panicking encode** (`TryLockError::Poisoned` arm)
— that panic unwinds directly across the `unsafe extern "C" fn`
trampoline `rb_thread_call_without_gvl` (today) or `rb_nogvl` (after this
change) calls into. Unwinding across an `extern "C"` boundary compiled
without `-C panic=abort` is undefined behavior in Rust; it does not become
*more* likely under `rb_nogvl`, but it also isn't fixed by switching to it.
This should be closed — by wrapping the closure body in
`std::panic::catch_unwind` inside `call_without_gvl` and converting a caught
panic into the same `Result`/`raise(ruby, ...)` path `tokenizer.rs` already
uses for ordinary errors — as part of whichever iteration lands this change,
since `ext/gigatoken/**` is out of this spike's boundary to fix directly.
See DISAGREEMENTS.

**Ruby/Async version floor.** `blocking_operation_wait` and
`RB_NOGVL_OFFLOAD_SAFE` are Ruby 3.4+/Async v2.21+ features
(`~/architect/src/github.com/socketry/async/lib/async/scheduler.rb:55-56`,
"@public Since *Async v2.21* and *Ruby v3.4*"; release note at
`~/architect/src/github.com/socketry/async/releases.md:382`, "Ruby 3.4 will
feature a new fiber scheduler hook"). `gigatoken.gemspec` currently floors at
`>= 3.3.0`. Hand-declaring the flag constant (as `gvl.rs` already does for
`rb_thread_call_without_gvl`, avoiding an `rb-sys` dependency — see its
top-of-file comment) means this compiles regardless of Ruby version; the
open question is runtime behavior of an unrecognized flag bit on Ruby 3.3,
which needs verifying against 3.3 before merging (this worktree only has
Ruby 4.0.6 installed — `ruby -v` → `ruby 4.0.6 (2026-07-14 revision
03b6d3f889) +PRISM [arm64-darwin27]`). See DISAGREEMENTS.

## Rejected alternatives

- **FD/pipe/eventfd completion** (the brief's candidate #1): rejected as
  strictly more machinery for the same outcome. It would need: a
  hand-rolled background-thread spawn (not `rb_thread_create`, so it doesn't
  get `io-event`'s GC-safe thread bookkeeping for free), a pipe/eventfd pair
  with manual fd lifecycle (who closes it on early GC, on error, on Ruby
  process fork), a side channel to move the actual `(Vec<u32>, Vec<i64>)`
  result across (the pipe itself can only signal readiness, not carry a
  `Vec<u32>` payload without extra unsafe plumbing), and manual panic-safety
  on that background thread so a Rust panic doesn't leave the waiting fiber
  parked on `io.wait_readable` forever. `rb_nogvl(...,
  RB_NOGVL_OFFLOAD_SAFE)` gets fiber-cooperative completion, cross-thread
  fiber wakeup (`rb_fiber_scheduler_unblock`), and cancellation plumbing
  (`rb_fiber_scheduler_blocking_operation_cancel`) from Ruby/io-event for
  free, with a one-flag change to existing code.
- **A new `encode_files_async` method returning something to await inside
  `Async { }`**: rejected — the brief's own sketch implies new Ruby-visible
  API surface, but `rb_nogvl(OFFLOAD_SAFE)` makes the *existing* synchronous
  `encode_batch`/`encode_files` already fiber-cooperative. Adding a second,
  differently-named method would duplicate the sync path for no behavioral
  gain and would need its own tests/docs/back-compat story that the
  single-flag change avoids entirely.
- **Manually calling `rb_fiber_scheduler_blocking_operation_wait` directly**
  (the raw C API behind `rb_nogvl`'s flag, `ruby/fiber/scheduler.h:478`):
  rejected — `rb_nogvl` is the documented, stable-looking entry point that
  already does the "is there a scheduler with this hook, if not fall back"
  branch internally; calling the lower-level function ourselves would just
  re-implement that branch for no benefit, and the header marks several of
  the primitives around it (`rb_fiber_scheduler_blocking_operation_extract`,
  `_execute`, `_cancel`) `@note Experimental`.
- **Spawning our own Ruby Thread to host the encode**: rejected outright,
  independent of mechanism — this project's standing rule is fibers, never
  threads, in our own code (`~/.claude/CLAUDE.md`); `io-event`'s
  `WorkerPool` already does this for us as the socketry-internal
  multi-reactor-hosting exception the same rule carves out.

## Follow-up iteration scope

1. In `ext/gigatoken/src/gvl.rs`: swap `rb_thread_call_without_gvl` for
   `rb_nogvl(..., RB_NOGVL_OFFLOAD_SAFE)` per the sketch above. No signature
   change to `without_gvl`, so `tokenizer.rs` needs no edits.
2. Wrap the closure call in `call_without_gvl` with
   `std::panic::catch_unwind`, converting a caught panic to a
   `Gigatoken::Error` via the existing `error::raise` path instead of
   unwinding across the C trampoline (closes the pre-existing gap above).
3. Verify `RB_NOGVL_OFFLOAD_SAFE`'s runtime behavior on the gemspec's Ruby
   3.3 floor (this worktree only has 4.0.6); either confirm it's a
   harmless no-op flag bit pre-3.4, or gate the flag value behind a
   `RUBY_VERSION` check.
4. Add a README/CHANGELOG note: `encode_batch`/`encode_files` only avoid
   stalling the reactor when the active `Fiber.scheduler` has a worker pool
   (`ASYNC_SCHEDULER_WORKER_POOL=true`, or an explicitly constructed
   `Async::Scheduler.new(worker_pool: ...)`); document the default
   single-worker queuing behavior for concurrent encodes.
5. A benchmark comparing encode-under-`Async` latency/throughput with the
   worker pool on vs. off (per Async's own release-note caveat that it
   isn't a guaranteed net win), sized against `benches/encode*.rs`'s
   existing harness.
6. Not in scope for the follow-up: real mid-encode cancellation (today's
   `ubf: None` means a fiber interrupt/timeout waits for the in-flight
   encode to finish rather than aborting it — unchanged by this
   recommendation, and follows the existing `without_gvl` contract).

## Experiment log

No throwaway magnus/Rust experiments were built for this spike: the
mechanism (`rb_nogvl` + `RB_NOGVL_OFFLOAD_SAFE`) and its integration with
`Async::Scheduler`/`IO::Event::WorkerPool` are fully specified and
implemented in the vendored source read above (Ruby 4.0.6's own
`ruby/thread.h` header, `async` 2.43.0, `io-event` 1.19.3, all installed in
this worktree's environment), so the design question was answerable by
reading real, load-bearing source rather than by prototyping. The one open
item that *would* need a real toy program to close is item 3 above (`RB_NOGVL_OFFLOAD_SAFE`
on Ruby 3.3), which needs a Ruby 3.3 interpreter this environment doesn't
have installed (`ruby -v` here is 4.0.6 only).
