---
type: how-to
---

# Work with packed results

`encode_batch` and `encode_files` return one Ruby Array per document by
default. For a large batch that is thousands of Arrays and millions of
Integers to build and later collect. `packed: true` returns a
[`Gigatoken::PackedResult`](../reference/packed-result.md) instead: every
document's ids back to back in one `IO::Buffer`, plus each document's
length.

```ruby
packed = tok.encode_files("owt_train.txt", separator: "<|endoftext|>", packed: true)

packed.size          # documents
packed.token_count   # tokens across all of them
packed.lens          # => [12, 8, 41, ...]
```

The whole result costs a fixed handful of Ruby objects however many
documents it holds.

## Read one document

```ruby
packed[3]      # => Array of ids, built on demand, one object
packed[-1]     # negative indices count from the end; out of range gives nil
```

## Iterate

```ruby
packed.each { |ids| ... }           # Enumerable
packed.map(&:size) == packed.lens   # => true
packed.to_a                         # the ragged shape encode_batch would have returned
```

Each document you touch costs one Array; documents you skip cost nothing.

## Hand the buffer to something else

```ruby
packed.buffer                        # a read-only IO::Buffer of u32, native byte order
packed.buffer.get_string             # the raw bytes, if you need them as a String
packed.buffer.values(:u32, 0, 8)     # the first eight ids, straight from the buffer
```

Offsets are in bytes (four per token). The buffer is read-only — writing to
it raises `IO::Buffer::AccessError` — because it wraps the String the engine
gathered into, with no copy. Read-only is not the same as `frozen?`, which
is `false` for the buffer object itself.

## Related

- [`Gigatoken::PackedResult` reference](../reference/packed-result.md)
- [Allocations](../explanation/allocations.md) — what each shape costs
