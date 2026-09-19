---
type: reference
---

# Command line

The gem installs a `gigatoken` executable with two commands.

### `gigatoken bench TOKENIZER FILES... [options]`

Encodes FILES and reports seconds, MB/s and Mtok/s, mirroring upstream's
Python CLI.

| Option | Default | Effect |
| ------ | ------- | ------ |
| `--doc-separator SEP` | none | split files on SEP (e.g. `"<\|endoftext\|>"`); whole files are single documents otherwise |
| `--limit-bytes N` | `none` | cap the bytes benchmarked, e.g. `100MB` or `64MiB` (parallel mode only) |
| `--[no-]parallel` | parallel | `--no-parallel` runs the fused serial core path |
| `--packed` | off | time the fused native file path with a packed `IO::Buffer` result (ignores `--limit-bytes`) |
| `--pretokenizer NAME` | none | required when TOKENIZER is a `.tiktoken` file; ignored otherwise |

### `gigatoken validate TOKENIZER FILES... [options]`

Confirms the native split-and-encode agrees with a Ruby-side split fed
through `encode_batch`. Takes `--doc-separator` and `--pretokenizer` as
above.

TOKENIZER is anything `Gigatoken::Tokenizer.load` accepts: a
`tokenizer.json` path or directory, a packaged encoding name, a Hub repo id,
or a `.tiktoken` file (with `--pretokenizer`).

## Sizes

`--limit-bytes` takes decimal units (`KB`, `MB`, `GB`, `TB` — powers of
1000) or binary ones (`KiB`, `MiB`, `GiB`, `TiB` — powers of 1024); a bare
number is bytes, and `none` or `unlimited` means no cap.

## Compressed FILES

`.gz` files work in both commands: the Ruby-side split decompresses them the
way the native file sources do, so `validate` compares like with like and
`bench` reports MB/s over the decompressed bytes.

`.zst` does not: gigatoken-rb has no Ruby-side zstd decoder (the `zstd-ruby`
gem is not a dependency), and reporting throughput over compressed bytes or
validating decompressed output against compressed input would both be
silently wrong, so both commands refuse it with `error: ...`. Decompress the
file first, or use the library's `Tokenizer#encode_files`, which handles
`.zst` natively.

## Errors

Errors print a single `error: ...` line and exit 1, with no backtrace —
a tokenizer that will not load, a missing or unreadable FILE, an empty
`--doc-separator`, or an empty FILES list.

```sh
gigatoken bench cl100k_base owt_train.txt --doc-separator "<|endoftext|>" --packed
gigatoken bench lib/gigatoken/encodings/cl100k_base.tiktoken README.md --pretokenizer gpt4
gigatoken validate openai-community/gpt2 owt_train.txt --doc-separator "<|endoftext|>"
```
