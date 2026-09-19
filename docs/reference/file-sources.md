---
type: reference
---

# File sources

The `source` argument of `Tokenizer#encode_files`: paths plus how to split
their bytes into documents. Three native classes, one per format. All take
an Array of paths (Strings or `Pathname`s). `.gz` and `.zst` files are
decompressed transparently.

### `Gigatoken::Native::TextFileSource.new(paths, separator: nil)`

Plain text. With `separator:` (a String of bytes), documents are the pieces
between its occurrences across each file; without one, each file is a
single document. `Tokenizer#encode_files` builds this class for you when
given a bare path or an Array of paths.

### `Gigatoken::Native::JsonlFileSource.new(paths, field: "text")`

JSON Lines: one document per line, text taken from `field`.

### `Gigatoken::Native::ParquetFileSource.new(paths, column: "text")`

Parquet: one document per row, text taken from `column`. Null cells become
empty documents.

## Behaviour

- Text and JSONL files are memory-mapped (or decompressed into memory) and
  handed to the fused split-and-encode core as byte regions; Parquet rows
  are read as owned documents. Nothing becomes a Ruby String.
- With `parallel: true` (the default) the work runs on the engine's worker
  pool; with `parallel: false` on the calling thread, same output.
- SentencePiece tokenizers require file contents and any text `separator:`
  to be valid UTF-8; otherwise `Gigatoken::Error`.
- A missing file raises `Gigatoken::Error` with the OS error.

## Files must not change under a running `encode_files`

An uncompressed file is memory-mapped, not copied, so the worker threads
read the file itself for the whole call. Truncating it or rewriting it in
place while `encode_files` runs faults those pages and the process receives
**SIGBUS** — a signal Ruby cannot rescue, so the process dies mid-call; no
`begin`/`rescue` around `encode_files` can save it.

Rotate by rename, never by truncate: write the replacement to a new path and
`File.rename` it over the old one. The mapping keeps the original inode
alive until the call finishes, and the next call picks up the new file.
Appending to a mapped file is safe (the mapping simply does not see the new
bytes); shrinking it is not.

Compressed inputs are exempt: `.gz` and `.zst` files are decompressed into
memory up front, so nothing maps the file after the read completes.

## Related

- [Tokenize files without leaving Rust](../how-to/tokenize-files.md)
