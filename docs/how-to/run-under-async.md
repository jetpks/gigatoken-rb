# Async-cooperative encodes

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

The pool defaults to **one** background worker: a second concurrent encode
from another fiber queues behind the first rather than running in parallel.
Apps that want several encodes in flight at once should size the pool
themselves (`IO::Event::WorkerPool.new(maximum_worker_count: N)`).

See `docs/rb/async-design.md` for the full design/safety writeup and
`bench/async_heartbeat.rb` for a runnable proof that the calling fiber yields
only with the worker pool enabled.
