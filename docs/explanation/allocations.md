---
type: explanation
---

# Allocations: what a call costs, and why the seam is shaped this way

The engine does its work in Rust. What this gem adds on top is a seam:
marshaling Ruby Strings in and token ids out. Every object that seam
allocates is one more thing for Ruby's GC to mark and sweep on every
collection for as long as it lives, so the seam is designed to allocate
only what it returns. This page says what each operation costs in Ruby
objects, why, and where the remaining costs come from.

## Per operation, on Ruby 4.0.7

| Operation | Ruby objects | What they are |
|---|---|---|
| `encode(text)` | 1 | the Array of ids (Integers are immediates) |
| `decode(ids)` | 1 | the String |
| `encode_batch(docs)` | n + 2 | one Array per document, the outer Array, a snapshot of the input Array |
| `encode_batch(docs, packed: true)` | 9 | the `IO::Buffer`, the String it wraps, the lens Array, the `PackedResult` and its offsets, the input snapshot, the pair the extension returns and its splat |
| `PackedResult#[]` | 1 | the Array of ids |
| `PackedResult#to_a` | n + 1 | one Array per document plus the outer one |
| `encode_files(path, packed: true)` | 16 | as the packed batch, plus the `TextFileSource` and the argument ceremony around it |
| `Tokenizer.from_encoding(name)` | 2 | the `Tokenizer` and its native tokenizer |
| `Tokenizer.load(name)` | 3 | the same, plus the path copy `File.exist?` makes while deciding the source's shape |

The budgets are frozen in `spec/gigatoken/allocations_spec.rb`, so a
change that adds an object fails the suite.

## Why the shapes are what they are

**Ragged results are one Array per document, and that is the floor.** A
Ruby Array of Integer ids is what callers asked for; the extension builds
each one straight from the engine's flat output slice, with no
intermediate copy on the Rust side, and never materializes a per-token
object because small Integers are immediates.

**Packed results are the way out of even that.** A million documents is a
million Arrays for the GC to walk. `packed: true` hands back one
`IO::Buffer` over the String the engine gathered into — zero-copy, frozen,
read-only — plus a lens Array, and unpacks a document only when you index
it. On the batch path the engine writes token ids straight into that
String's buffer while the GVL is released; the String is allocated with the
GVL held, sized to the worst case (one token per input byte), and trimmed
to what was written. That reservation is the packed path's malloc column:
four bytes per input byte, briefly.

**Inputs are borrowed, not copied, when that is sound.** A heap String
(one whose bytes live outside its RVALUE) that is frozen is read in place
during a batch encode; one that isn't frozen is locked against mutation
for the call and read in place; an embedded String (under about a
kilobyte on Ruby 4.0) is copied, because its bytes live inside an object
the GC may move. The call snapshots the input Array first, so the
documents encoded are the ones the array held when the call began,
whatever a `to_str` conversion or another thread does to it meanwhile.

**Construction builds only what the source needs.** `Tokenizer.load`
dispatches on the shape of its argument; the HuggingFace client, an
`async-http` internet object, is created only when the argument turns out
to be a repo id.

**Reading a packed document is one Array.** `IO::Buffer#values(:u32,
offset, count)` returns the ids directly. (Its sibling `get_values` takes
the layout as an Array of type symbols, which used to be built per call,
as long as the document.)

## What does not show in these counts

Rust-side allocations. The extension routes them through mimalloc, and
Ruby's `GC.stat` counters see neither them nor the engine's own buffers.
They cost time, not GC work; `bench/operations.rb`'s time column and
`bench/encode_ab.rb` are how a change there is measured. The 0.3.0 removal
of the per-document copy in the ragged path is the kind of change that is
invisible here and visible there.

## Related

- [Benchmarks](benchmarks.md) — the time column, and the corpus numbers
- [Work with packed results](../how-to/use-packed-results.md)
- [Measure time and allocations](../how-to/measure-time-and-allocations.md)
