---
type: how-to
---

# Tokenize files without leaving Rust

`Tokenizer#encode_files` reads, splits and encodes files on the native side
in one fused pass. The documents never become Ruby Strings; only the token
ids come back.

## Plain text, one document per separator

```ruby
tok.encode_files("owt_train.txt", separator: "<|endoftext|>")
tok.encode_files(["part1.txt", "part2.txt"], separator: "\n\n")
tok.encode_files("whole.txt")     # no separator: each file is one document
```

Bare paths are wrapped in a `Gigatoken::Native::TextFileSource`. `.gz` and
`.zst` files are decompressed transparently.

## JSON Lines and Parquet

```ruby
jsonl = Gigatoken::Native::JsonlFileSource.new(["docs.jsonl"], field: "text")
tok.encode_files(jsonl)

parquet = Gigatoken::Native::ParquetFileSource.new(["docs.parquet"], column: "text")
tok.encode_files(parquet)
```

`field:` and `column:` default to `"text"`. Null Parquet cells become empty
documents.

## Choose the result shape

```ruby
ragged = tok.encode_files(source)                 # Array of Arrays, one per document
packed = tok.encode_files(source, packed: true)   # one IO::Buffer plus per-document lengths
```

`packed: true` skips the per-document Ruby Arrays; see
[Work with packed results](use-packed-results.md).

## Stay off the worker pool

```ruby
tok.encode_files(source, parallel: false)
```

Loads and encodes everything on the calling thread, with identical output.
Useful when the process already saturates its cores or you want a
deterministic single-threaded run to compare against.

## SentencePiece models

The SentencePiece backend decodes text, so file contents and a text
`separator:` must be valid UTF-8; invalid bytes raise `Gigatoken::Error`
instead of being guessed at.

## Related

- [File sources reference](../reference/file-sources.md)
- [Run encodes under Async](run-under-async.md) — the whole call releases the GVL
