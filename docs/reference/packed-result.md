---
type: reference
---

# `Gigatoken::PackedResult`

The result of `encode_batch`/`encode_files` with `packed: true`: every
document's token ids back to back in one `IO::Buffer`, plus per-document
lengths. Documents are unpacked on demand. Includes `Enumerable`.

Constructed by the tokenizer; `Gigatoken::PackedResult.new(buffer, lens)`
is public for tests and for wrapping a buffer you built yourself.

## Attributes

### `#buffer → IO::Buffer`

Read-only, `u32` token ids in native byte order, four bytes per token, no
padding between documents. Wraps the String the engine gathered into
without copying it.

### `#lens → Array<Integer>`

Token count of each document, in order.

## Counting

### `#size → Integer`

Number of documents.

### `#token_count → Integer`

Total tokens across every document.

## Reading documents

### `#[](i) → Array<Integer> | nil`

Document `i`'s ids as a new Array, read straight from the buffer
(`IO::Buffer#values`). Negative indices count from the end; an index out
of range returns `nil`. One object per call.

### `#each { |ids| } → self`, `#each → Enumerator`

Yields each document's ids in order, one Array per document.

### `#to_a → Array<Array<Integer>>`

The ragged shape `encode_batch`/`encode_files` return without
`packed: true`; one Array per document plus the outer Array.

## Examples

```ruby
packed = tok.encode_batch(["Hello, world!", "Another document."], packed: true)
packed.size          # => 2
packed.lens          # => [4, 3]
packed.token_count   # => 7
packed[1]            # => [14364, 2246, 13]
packed[-1]           # => [14364, 2246, 13]
packed[2]            # => nil
packed.to_a          # => [[9906, 11, 1917, 0], [14364, 2246, 13]]
packed.buffer.values(:u32, 0, 4)   # => [9906, 11, 1917, 0]
```

## Related

- [Work with packed results](../how-to/use-packed-results.md)
- [`Gigatoken::Tokenizer`](tokenizer.md)
