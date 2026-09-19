---
type: how-to
---

# Tune the encode-cache budget

Each tokenizer keeps a cache of pretoken → ids results on its
single-document `encode` path, and batch encodes keep one per worker. Left
unbounded, a long-lived process (a Rails worker, a Sidekiq job) that sees
ever-new text grows it without limit, so the cache is capped process-wide:
512 MiB per worker by default. When a cache hits its budget it wipes back
toward its vocabulary seed and refills. Output never changes; only some
recomputation is paid.

## Read and set the budget

```ruby
Gigatoken.max_cache_bytes            # => 536870912 (512 MiB)
Gigatoken.max_cache_bytes = 64 << 20 # 64 MiB per worker
Gigatoken.max_cache_bytes = nil      # unbounded
```

The budget applies to tokenizers built *after* it is set. A tokenizer that
already exists keeps the budget it was built with.

## Size it

A parallel batch encode may use up to `workers × budget`, where `workers`
is the engine's pool size (the machine's core count). On a 16-core box the
default allows 8 GiB of cache in the worst case; a memory-constrained
container should set the budget accordingly before loading tokenizers.

## Watch it

```ruby
tok.cache_entries    # cached pretoken/unit entries on the single-document path right now
```

Grows as text is encoded, and drops back toward the vocabulary-seed level
after a wipe.

## Related

- [Encodings, settings and errors](../reference/encodings-and-settings.md)
