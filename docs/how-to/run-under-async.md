---
type: how-to
---

# Run encodes under Async

`Tokenizer#encode_batch` and `#encode_files` release the GVL via
`rb_nogvl(..., RB_NOGVL_OFFLOAD_SAFE)` (`ext/gigatoken/src/gvl.rs`). No new
API: the same synchronous calls you already use.

```ruby
Async do
  tokenizer.encode_files(source) # other fibers in this Async block keep
end                              # running while this encode is in flight
```

That only actually yields the calling **fiber** — not just other Threads —
when the active `Fiber.scheduler` was built with a worker pool:

```
ASYNC_SCHEDULER_WORKER_POOL=true ruby my_app.rb
```

or explicitly:

```ruby
Fiber.set_scheduler(Async::Scheduler.new(worker_pool: IO::Event::WorkerPool.new))
```

Without a worker pool — a bare `Async { ... }`, or no scheduler at all —
`encode_batch`/`encode_files` block exactly as before: the GVL is released
(other Ruby Threads can run) but the calling fiber does not yield to the
reactor.

The pool is a **Ruby 4.0** feature: io-event builds `IO::Event::WorkerPool`
only against 4.0's blocking-operation API, so on Ruby 3.3 and 3.4
`ASYNC_SCHEDULER_WORKER_POOL=true` changes nothing — the encode blocks the
calling fiber, and an `Async` timeout around it fires only once the batch
returns. `Timeout.timeout`, `Thread#kill` and Ctrl-C cancel the encode on
every supported Ruby.

The pool defaults to **one** background worker: a second concurrent encode
from another fiber queues behind the first rather than running in parallel.
Apps that want several encodes in flight at once should size the pool
themselves (`IO::Event::WorkerPool.new(maximum_worker_count: N)`).

## Timeouts cancel the encode

A timeout around a batch cancels it rather than waiting for it:

```ruby
Async do |task|
  task.with_timeout(0.5) { tokenizer.encode_batch(docs) }
rescue Async::TimeoutError
  # raised within a document of the deadline, not when the batch would
  # have finished; `docs` is untouched and the tokenizer is reusable
end
```

The scheduler cancels the offloaded operation, the encode stops at the next
document boundary and its partial result is discarded — with the worker pool,
so on Ruby 4.0 (above). The same holds for `Timeout.timeout`, `Thread#kill`
and Ctrl-C outside Async, on every supported Ruby. SentencePiece
batches are the exception: they finish the encode first and raise after —
safely, just not early.

See [Async design](../explanation/async-design.md) for the full design and
safety writeup, and `bench/async_heartbeat.rb` for a runnable proof that the
calling fiber yields only with the worker pool enabled.
