---
type: reference
---

# Encodings, settings and errors

## `Gigatoken::Encodings`

The tiktoken encodings vendored inside the gem, with the pieces a
`.tiktoken` file doesn't carry. Provenance, source URLs and hashes are in
`lib/gigatoken/encodings/PROVENANCE.md`.

### `Gigatoken::Encodings::NAMES → Array<String>`

`["r50k_base", "cl100k_base", "o200k_base", "o200k_harmony"]`.

### `Gigatoken::Encodings[name] → Hash | nil`

`{rank_file:, pretokenizer:, special_tokens:}` for a packaged name, or
`nil`. `o200k_harmony` reuses `o200k_base`'s ranks and scheme and differs
only in its special-token table (10 named control tokens plus 1081
reserved slots).

### `Gigatoken::Encodings.unpackable_reason(name) → String | nil`

Why a known name isn't packaged (`p50k_base`, `p50k_edit`: non-dense
ranks), or `nil`.

## Pretokenizer schemes

### `Gigatoken::Native.pretokenizer_names → Array<String>`

The scheme names `from_tiktoken`/`load` accept: `gpt2`, `r50k`, `gpt4`,
`cl100k`, `o200k`, `qwen2`, `qwen35`, `olmo3`, `deepseek_v3`, `nemotron`,
`kimi` (aliases share a scheme). The single source of the list the
"unknown scheme" error prints.

## Cache budget

### `Gigatoken.max_cache_bytes → Integer | nil`
### `Gigatoken.max_cache_bytes = bytes`

Process-global encode-cache budget in bytes per worker, applied to
tokenizers built afterwards. `nil` means unbounded. Default 512 MiB. See
[Tune the encode-cache budget](../how-to/tune-the-cache-budget.md).

## Errors

### `Gigatoken::Error < StandardError`

Every load and encode failure surfaced from the native extension: unknown
scheme, unpackaged encoding, missing file, invalid UTF-8 on the
SentencePiece path, Hub HTTP errors. A Rust panic never crosses the
boundary as anything else.

`ArgumentError`/`TypeError` come from Ruby argument handling as usual (a
non-String element in `encode_batch`, a bad `packed:` value).

## Related

- [`Gigatoken::Tokenizer`](tokenizer.md)
- [Load a tokenizer from any source](../how-to/load-a-tokenizer.md)
