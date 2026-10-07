# Vendored encodings — provenance

The `.tiktoken` files in this directory are OpenAI's published BPE mergeable-rank
tables, vendored verbatim so `gigatoken` can resolve these encodings by name with
no network access and no writable cache directory.

Each file is a plain text table: one `base64(token_bytes) rank` pair per line. It
carries **mergeable ranks only** — the pretokenizer split regex and the special
tokens belong to the encoding's *definition*, not to the file, and live in code
(see the table below).

The three `.json.gz` files are HuggingFace `tokenizer.json` files, not tiktoken
tables; they are covered in [their own section](#huggingface-tokenizerjson-files)
at the end.

They live under `lib/` rather than a top-level `data/` directory because this
repo's `.gitignore` ignores `/data/` ("downloaded test data"); these are shipped
gem payload, not test fixtures.

## Files

| File | Bytes | sha256 |
|------|------:|--------|
| `r50k_base.tiktoken`   |   835,554 | `306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930` |
| `cl100k_base.tiktoken` | 1,681,126 | `223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7` |
| `o200k_base.tiktoken`  | 3,613,922 | `446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d` |

## Source

Retrieved **2026-08-10** over HTTPS from OpenAI's public encodings endpoint:

```
https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken
https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken
```

These are the same URLs `openai/tiktoken` itself fetches from, in
[`tiktoken_ext/openai_public.py`](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py).

## Authenticity

Verified three independent ways at retrieval time:

1. **Transport** — HTTPS directly from `openaipublic.blob.core.windows.net`, the
   origin OpenAI publishes and `tiktoken` itself downloads from.
2. **Publisher checksum** — each measured sha256 above matches the `expected_hash`
   OpenAI publishes for that file in `openai_public.py`, fetched separately from
   `github.com/openai/tiktoken`. Two independent channels agree on the bytes.
3. **Behavioral, against a third-party implementation** — loaded through
   `gigatoken` with the pretokenizer and special tokens below, every token id
   matched [`tiktoken_ruby`](https://github.com/IAPark/tiktoken_ruby) 0.0.17
   (which embeds its own copy of these ranks) across a corpus covering ASCII,
   CJK, ZWJ emoji sequences, combining accents, whitespace runs, source code,
   and URLs. Resulting `vocab_size`: 50257 / 100277 / 200019.

## Encoding definitions

Transcribed from `openai_public.py` (each encoding's `pat_str` and
`special_tokens`), cross-checked against upstream gigatoken's own port in
`gigatoken/_load/tiktoken.py`. The scheme names are `PretokenizerType::NAMES`
values (`src/pretokenize/options.rs`).

| Encoding | Pretokenizer scheme | Special tokens |
|---|---|---|
| `r50k_base`   | `gpt2`  | `<\|endoftext\|>`=50256 |
| `cl100k_base` | `gpt4`  | `<\|endoftext\|>`=100257, `<\|fim_prefix\|>`=100258, `<\|fim_middle\|>`=100259, `<\|fim_suffix\|>`=100260, `<\|endofprompt\|>`=100276 |
| `o200k_base`  | `o200k` | `<\|endoftext\|>`=199999, `<\|endofprompt\|>`=200018 |

`p50k_base` is deliberately absent: its ranks are not dense (50256 is left free
for `<|endoftext|>`), and the rank loader rejects non-dense ranks with
`"ranks must be dense"`. `p50k_edit` loads the same `p50k_base.tiktoken` ranks
and is absent for the identical reason.

## `o200k_harmony`

Packaged with **no new vendored file**: it reuses `o200k_base.tiktoken`'s
mergeable ranks and the `o200k` pretokenizer scheme verbatim. Confirmed
against `openai/tiktoken` 0.9.0 that `o200k_harmony()` in `openai_public.py`
calls `mergeable_ranks` with the same rank file and uses the same `pat_str` as
`o200k_base()` — only the special-token table differs.

That table (1091 entries: 10 named control tokens, 1081
`<|reserved_N|>` slots) is transcribed, not derived, from `o200k_harmony()`:
base specials `<|endoftext|>` 199999 and `<|endofprompt|>` 200018, then
`<|startoftext|>` 199998, `<|endoftext|>` 199999, `<|reserved_200000|>`
200000, `<|reserved_200001|>` 200001, `<|return|>` 200002, `<|constrain|>`
200003, `<|reserved_200004|>` 200004, `<|channel|>` 200005, `<|start|>`
200006, `<|end|>` 200007, `<|message|>` 200008, `<|reserved_200009|>` 200009,
`<|reserved_200010|>` 200010, `<|reserved_200011|>` 200011, `<|call|>`
200012, then `<|reserved_N|>` for `N` in `200013..201087`. The reserved range
is **not** contiguous from 200000 — the named control tokens sit inside
200000..200012, leaving reserved slots only at `{200000, 200001, 200004,
200009, 200010, 200011} ∪ [200013, 201087]`. A table built as "200000..201087
minus the named ids" invents `<|reserved_200002|>` and drops
`<|reserved_200018|>`; the transcription above avoids both.

**Not verified against `tiktoken_ruby`** (unlike the three encodings above):
`tiktoken_ruby` 0.0.17's own `o200k_harmony` table drops `<|endofprompt|>` —
it encodes the literal as six ordinary-text tokens rather than `[200018]` —
while treating `<|reserved_200018|>` as the sole literal at that id.
`openai/tiktoken` 0.9.0 keeps both `<|endofprompt|>` and
`<|reserved_200018|>`, at the same id, as Python dict construction preserves
both keys. `openai/tiktoken` is authoritative here; `tiktoken_ruby` is the
outlier. See `spec/gigatoken/differential_spec.rb` for the measurement and
the reduction-plus-pinning proof used in place of a `tiktoken_ruby`
comparison.

## Licence

The `tiktoken` project and its published encoding files are MIT licensed,
Copyright (c) 2022 OpenAI, Shantanu Jain. See
<https://github.com/openai/tiktoken/blob/main/LICENSE>. The files are vendored
here unmodified.

## HuggingFace `tokenizer.json` files

`qwen35.json.gz`, `qwen38.json.gz` and `muse_spark.json.gz` are HuggingFace
`tokenizer.json` files, vendored byte-for-byte and gzipped. Unlike a
`.tiktoken` table, a `tokenizer.json` carries the encoding's whole definition
— vocabulary, merges, normalizer, pretokenizer, added tokens — so the registry
records only the file, and `Tokenizer.from_encoding` loads it with
`Tokenizer.from_json`, reading the special tokens from the file's
`added_tokens` (those marked `"special": true`) exactly as `from_json` does.

### Files

| File | Decompressed bytes | sha256 (decompressed) |
|------|-------------------:|-----------------------|
| `qwen35.json.gz`     | 12,807,982 | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| `qwen38.json.gz`     | 12,809,320 | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| `muse_spark.json.gz` | 28,129,897 | `c9dbee66967b58f31a7c27f723c3760da3526ccd0427578e8905b0abb0031c4d` |

### Source

Retrieved **2026-10-07** over HTTPS from the Hub's `resolve` endpoint, each at
a pinned commit:

| Encoding | Repo | Revision | URL |
|---|---|---|---|
| `qwen35`     | `Qwen/Qwen3.5-27B`             | `fc05daec18b0a78c049392ed2e771dde82bdf654` | <https://huggingface.co/Qwen/Qwen3.5-27B/resolve/fc05daec18b0a78c049392ed2e771dde82bdf654/tokenizer.json> |
| `qwen38`     | `Qwen/Qwen3.8-27B`             | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | <https://huggingface.co/Qwen/Qwen3.8-27B/resolve/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer.json> |
| `muse_spark` | `meta-models/Muse-Glimmer-30B` | `a4e59da52a7bc87ae7251dd5545c0dd437c44b68` | <https://huggingface.co/meta-models/Muse-Glimmer-30B/resolve/a4e59da52a7bc87ae7251dd5545c0dd437c44b68/tokenizer.json> |

**`qwen35` is shared across releases.** The Hub's tree API
(`https://huggingface.co/api/models/<repo>?blobs=true`) reports the identical
LFS sha256 `5f9e4d49…` for `tokenizer.json` in `Qwen/Qwen3.5-{0.8B, 2B, 4B, 9B,
27B, 35B-A3B, 122B-A10B, 397B-A17B}` and `Qwen/Qwen3.6-{27B, 35B-A3B}`
(queried 2026-10-07), so one file serves Qwen 3.5 and 3.6.
`Qwen/Qwen3.6-Plus` is gated and was not checked.

**`qwen38` ships its own file**, identical (`0997f410…`) across
`Qwen/Qwen3.8-{27B, Flash-Next, 2.4T-A95B}` and their `-FP8` twins. It differs
from `qwen35` only in seven more `"special": true` added tokens, ids
248070–248076: `<|audio_start|>`, `<|audio_end|>`, `<tts_pad>`,
`<tts_text_bos>`, `<tts_text_eod>`, `<tts_text_bos_single>`, `<|audio_pad|>`.
The vocabulary, merges, normalizer, pretokenizer, decoder and post-processor
are identical, so ordinary text encodes the same under both; only those seven
literals, `vocab_size` and `special_tokens` differ. It is vendored as a second
file rather than derived from the first because the engine reads added tokens
from the JSON, and a JSON-backed tokenizer has no way to add specials after
the fact. `Qwen3.8-27B` is the repo cited; `Flash-Next` and `2.4T-A95B` carry an
"other" licence on the *model*, but their tokenizer file is byte-identical to
the Apache-2.0 repo's.

**`muse_spark` is the Muse-Glimmer-30B file.** Meta publishes no standalone
Muse Spark tokenizer (a Hub search over `meta-models` finds none, and Meta's
model-API docs at <https://dev.meta.ai/docs/muse-glimmer> point at none). The
Muse-Glimmer-30B model card states "Tokenizer: 200,000 BPE tokens + 2,048
special tokens" (vocabulary 202,048), trained from Muse Spark's outputs, and
this exact repo and revision is the one the reference application pins. As of
2026-10-07 that revision is also `main`.

### Authenticity

Verified three independent ways at retrieval time:

1. **Transport** — HTTPS directly from `huggingface.co`, at the pinned
   commit above.
2. **Publisher checksum** — each file's sha256 above matches the LFS sha256 the
   Hub's tree API reports for `tokenizer.json` at that revision, and so does
   its size. Two independent channels agree on the bytes.
3. **Behavioral, against the reference implementation** — loaded through
   `Tokenizer.from_encoding`, every token id matched HuggingFace
   [`tokenizers`](https://rubygems.org/gems/tokenizers) 0.7.0 loading the same
   decompressed file with `add_special_tokens: false`, across this repo's own
   source and docs (`spec/gigatoken/differential_spec.rb`). Resulting
   `vocab_size`: 248070 / 248077 / 202048; `special_tokens.size`: 14 / 21 / 2048.

### Compression

Each file was compressed with `gzip -9 -n` (no name or timestamp in the gzip
header, so the bytes are reproducible), and `gunzip -c` of it reproduces the
published file byte-for-byte — the sha256s above are of the decompressed
bytes. Compressed sizes: 3,514,616 (`qwen35`), 3,514,699 (`qwen38`), 4,475,666
(`muse_spark`).

### What gigatoken does not take from the file

- **The post-processor.** gigatoken applies none. `muse_spark`'s is a
  `TemplateProcessing` that prepends `<|begin_of_text|>`; `Tokenizer#encode`
  does not, matching `tokenizers`' `add_special_tokens: false`. The Qwen files'
  is a `ByteLevel` post-processor, a no-op on ids.
- **Non-special added tokens in `special_tokens`.** Qwen's 12 non-special
  added tokens (`<tool_call>`, `<think>`, the `<|fim_*|>` tokens,
  `<|repo_name|>`, `<|file_sep|>`, `<tool_response>` and so on) are matched
  atomically by the engine, like the special ones, but only `"special": true`
  tokens appear in `Tokenizer#special_tokens` — 14 for `qwen35`, 21 for
  `qwen38`.
- **`tokenizer_config.json`.** Not vendored; gigatoken reads only
  `tokenizer.json`.

### Licence

All three repos are Apache-2.0, each with its own `LICENSE` at the pinned
revision:

- <https://huggingface.co/Qwen/Qwen3.5-27B/blob/fc05daec18b0a78c049392ed2e771dde82bdf654/LICENSE>
- <https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/LICENSE>
- <https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/a4e59da52a7bc87ae7251dd5545c0dd437c44b68/LICENSE>

The files are vendored here unmodified (gzipped).
