---
type: reference
---

# `Gigatoken::Tokenizer`

A tokenizer: encode, batch encode, decode and vocabulary introspection over
a native `Gigatoken::Native::BPETokenizer` or
`Gigatoken::Native::SentencePieceTokenizer`. The backend is chosen at load
time from the model (`byte_fallback: true` selects SentencePiece); the
public surface is the same for both.

Everything here raises a subclass of `Gigatoken::Error`, never a raw Rust
panic: `Gigatoken::ModelError` when a tokenizer cannot be loaded,
`Gigatoken::InputError` when a document or an id cannot be taken, and
`Gigatoken::HubError` for anything [`Gigatoken::Hub`](encodings-and-settings.md#hub-requests)
raises on the way. `rescue Gigatoken::Error` still catches all of them.

## Constructors

### `Gigatoken::Tokenizer.load(source, pretokenizer: nil, special_tokens: {}, revision: "main", hub: nil)`

Dispatches on the shape of `source` (a String, or anything with `to_s`):

| Shape | Handled by |
| ----- | ---------- |
| ends in `.tiktoken` | `from_tiktoken` — `pretokenizer:` is required, `special_tokens:` optional |
| an existing file or directory | `from_file` |
| a packaged encoding name (`Gigatoken::Encodings::NAMES`) | `from_encoding` |
| a name the registry knows but doesn't package (`p50k_base`, `p50k_edit`) | raises `Gigatoken::ModelError` with the reason |
| `org/name`, or a bare legacy repo name | `from_hub`, with `revision:` and `hub:` |

Packaged names are checked before the Hub-repo shape. `hub:` is a
`Gigatoken::Hub`; when omitted one is built only if the Hub path is taken.

**Raises** `Gigatoken::Error` for a `.tiktoken` path without
`pretokenizer:`, and for a source that matches no shape.

### `Gigatoken::Tokenizer.from_encoding(name)`

One of the packaged encodings by name — a String or a Symbol — entirely
from the vendored files. **Raises** `Gigatoken::ModelError` naming the
packaged encodings otherwise.

### `Gigatoken::Tokenizer.from_file(path)`

A `tokenizer.json` path, or a directory containing one. Reads it in binary.
**Raises** `Gigatoken::ModelError` naming the path when there is no file
there, or no `tokenizer.json` in the directory.

### `Gigatoken::Tokenizer.from_json(data)`

In-memory `tokenizer.json` contents (a String in any encoding, or anything
with `to_str`). Special tokens are read from its `added_tokens` (those with
`"special": true`). **Raises** `Gigatoken::ModelError` for JSON that doesn't
parse, including nesting deep enough to threaten the native parser's stack,
and `TypeError` for an argument that isn't String-convertible.

### `Gigatoken::Tokenizer.from_tiktoken(path, pretokenizer:, special_tokens: {})`

A `.tiktoken` mergeable-ranks file. `pretokenizer:` is one of
`Gigatoken::Native.pretokenizer_names`; `special_tokens:` maps token
content to id. **Raises** `Gigatoken::ModelError` for an unknown scheme,
naming the valid ones, for non-dense ranks, and for a file that isn't
readable as mergeable ranks.

### `Gigatoken::Tokenizer.from_hub(repo_id, revision: "main", hub: nil)`

`tokenizer.json` from a HuggingFace Hub repo, served from the standard HF
cache and downloaded into it on a miss. `hub:` is a `Gigatoken::Hub`; when
omitted one is built for the call, against `HF_ENDPOINT` if set. See
[Load a tokenizer](../how-to/load-a-tokenizer.md) for token and cache
discovery. **Raises** `Gigatoken::HubError` for an HTTP status, a transport
failure, a timeout, or a value that fails the Hub's
[checks](encodings-and-settings.md#what-is-checked).

## Encoding

### Input encodings

`#encode` and `#encode_batch` honour the String's encoding tag. UTF-8,
US-ASCII and ASCII-8BIT reach the native call byte-wise: the first two
already are UTF-8 bytes, and binary is deliberately raw. Anything else —
ISO-8859-1, Windows-1252, UTF-16LE — is a real encoding whose bytes are not
the text's UTF-8 bytes, so it is transcoded to UTF-8 first and gives the
same ids as the same text read as UTF-8. The caller's String is never
modified.

Invalid bytes in a UTF-8-tagged String go through raw — a documented
difference from tiktoken, which rejects them. A transcode that cannot be
done at all raises `Gigatoken::InputError`: a dummy encoding with no
converter (UTF-7), bytes the tag doesn't allow, a character UTF-8 can't
hold.

### `#encode(text) → Array<Integer>`

Token ids for one String. Literal special-token strings in the text are
tokenized as their special token (tiktoken's `encode_with_special_tokens`
behaviour). Runs on the calling thread and never releases the GVL.
Allocates one object, the Array. **Raises** `Gigatoken::InputError` for a
String that cannot be transcoded (see [Input
encodings](#input-encodings)).

### `#encode_batch(texts, packed: false) → Array<Array<Integer>> | Gigatoken::PackedResult`

Token ids for each String in `texts` (elements may also respond to
`to_str`), encoded on the engine's worker pool with the GVL released. The
result reflects `texts` as it was when the call began. Heap Strings are
read in place while the encode runs, locked against mutation; embedded
(short) Strings are copied.

With `packed: true`, a [`Gigatoken::PackedResult`](packed-result.md)
instead of one Array per document.

**Raises** `Gigatoken::InputError` if any element cannot be transcoded (see
[Input encodings](#input-encodings)), and `TypeError` for an element that
is neither a String nor `to_str`-convertible.

### `#encode_files(source, separator: nil, parallel: true, packed: false) → Array<Array<Integer>> | Gigatoken::PackedResult`

Reads and encodes whole files on the native side with the GVL released for
the whole call. `source` is a [file source](file-sources.md), a path, or an
Array of paths; bare paths become a `TextFileSource` with `separator:`.
`parallel: false` loads and encodes on the calling thread with identical
output.

### `#decode(ids) → String`

The bytes for an Array of token ids, as an `ASCII-8BIT` String. **Raises**
`Gigatoken::InputError` naming the id when one is outside the vocabulary
(anything `>= #vocab_size`).

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

One instance may be shared across threads, in any combination of calls: no
path blocks on a lock while holding the GVL, so none of them can stall the
VM. Batch encodes, `#decode` and the vocabulary readers take no lock at all;
`#encode` has its own per-instance worker, which it takes without waiting and
— on the rare occasion another thread's `#encode` holds it — waits for with
the GVL released.

An interrupt that arrives while a batch is in flight — `Timeout`,
`Thread#kill`, `Interrupt`, an [Async](../how-to/run-under-async.md) timeout
— cancels the batch at the next document boundary and raises where you called
it. The partial result is discarded, and the input Strings and the tokenizer
are left exactly as they were. SentencePiece `#encode_batch`/`#encode_files`
are the exception: they run to completion and raise after (interrupted
safely, just not early).

## Related

- [`Gigatoken::PackedResult`](packed-result.md)
- [Encodings, settings and errors](encodings-and-settings.md)
