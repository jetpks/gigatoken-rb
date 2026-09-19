---
type: reference
---

# `Gigatoken::Tokenizer`

A tokenizer: encode, batch encode, decode and vocabulary introspection over
a native `Gigatoken::Native::BPETokenizer` or
`Gigatoken::Native::SentencePieceTokenizer`. The backend is chosen at load
time from the model (`byte_fallback: true` selects SentencePiece); the
public surface is the same for both.

## Constructors

### `Gigatoken::Tokenizer.load(source, pretokenizer: nil, special_tokens: {}, revision: "main", hub: nil)`

Dispatches on the shape of `source` (a String, or anything with `to_s`):

| Shape | Handled by |
| ----- | ---------- |
| ends in `.tiktoken` | `from_tiktoken` — `pretokenizer:` is required, `special_tokens:` optional |
| an existing file or directory | `from_file` |
| a packaged encoding name (`Gigatoken::Encodings::NAMES`) | `from_encoding` |
| a name the registry knows but doesn't package (`p50k_base`, `p50k_edit`) | raises `Gigatoken::Error` with the reason |
| `org/name`, or a bare legacy repo name | `from_hub`, with `revision:` and `hub:` |

Packaged names are checked before the Hub-repo shape. `hub:` is a
`Gigatoken::Hub`; when omitted one is built only if the Hub path is taken.

**Raises** `Gigatoken::Error` for a `.tiktoken` path without
`pretokenizer:`, and for a source that matches no shape.

### `Gigatoken::Tokenizer.from_encoding(name)`

One of the packaged encodings by name, entirely from the vendored files.
**Raises** `Gigatoken::Error` naming the packaged encodings otherwise.

### `Gigatoken::Tokenizer.from_file(path)`

A `tokenizer.json` path, or a directory containing one. Reads it in binary.

### `Gigatoken::Tokenizer.from_json(data)`

In-memory `tokenizer.json` contents (String, any encoding). Special tokens
are read from its `added_tokens` (those with `"special": true`).

### `Gigatoken::Tokenizer.from_tiktoken(path, pretokenizer:, special_tokens: {})`

A `.tiktoken` mergeable-ranks file. `pretokenizer:` is one of
`Gigatoken::Native.pretokenizer_names`; `special_tokens:` maps token
content to id. **Raises** `Gigatoken::Error` for an unknown scheme, naming
the valid ones, and for non-dense ranks.

### `Gigatoken::Tokenizer.from_hub(repo_id, revision: "main", hub: Hub.new)`

`tokenizer.json` from a HuggingFace Hub repo, served from the standard HF
cache and downloaded into it on a miss. See
[Load a tokenizer](../how-to/load-a-tokenizer.md) for token and cache
discovery.

## Encoding

### `#encode(text) → Array<Integer>`

Token ids for one String. Literal special-token strings in the text are
tokenized as their special token (tiktoken's `encode_with_special_tokens`
behaviour). Runs on the calling thread and never releases the GVL.
Allocates one object, the Array.

### `#encode_batch(texts, packed: false) → Array<Array<Integer>> | Gigatoken::PackedResult`

Token ids for each String in `texts` (elements may also respond to
`to_str`), encoded on the engine's worker pool with the GVL released. The
result reflects `texts` as it was when the call began. Heap Strings are
read in place while the encode runs, locked against mutation; embedded
(short) Strings are copied.

With `packed: true`, a [`Gigatoken::PackedResult`](packed-result.md)
instead of one Array per document.

### `#encode_files(source, separator: nil, parallel: true, packed: false) → Array<Array<Integer>> | Gigatoken::PackedResult`

Reads and encodes whole files on the native side with the GVL released for
the whole call. `source` is a [file source](file-sources.md), a path, or an
Array of paths; bare paths become a `TextFileSource` with `separator:`.
`parallel: false` loads and encodes on the calling thread with identical
output.

### `#decode(ids) → String`

The bytes for an Array of token ids, as an `ASCII-8BIT` String.

## Introspection

### `#vocab_size → Integer`

### `#vocab → Hash<Integer, String>`

Token id to its bytes (`ASCII-8BIT`), built fresh on each call.

### `#merges → Array<[String, String]>`

The merge table as byte pairs, built fresh on each call.

### `#special_tokens → Hash<String, Integer>`

The special-token table the tokenizer was loaded with.

### `#cache_entries → Integer`

Cached pretoken/unit entries on the single-document `encode` path right
now. See [Tune the encode-cache budget](../how-to/tune-the-cache-budget.md).

## Thread safety

One instance may be shared across threads. Batch encodes and readers never
exclude each other; `#encode` takes a short exclusive lock and, only when
it must wait on an in-flight batch, releases the GVL while it waits.

## Related

- [`Gigatoken::PackedResult`](packed-result.md)
- [Encodings, settings and errors](encodings-and-settings.md)
