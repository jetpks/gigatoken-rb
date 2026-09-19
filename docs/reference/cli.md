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
| `--limit-bytes N` | `none` | cap the bytes benchmarked, e.g. `100MB` (parallel mode only) |
| `--[no-]parallel` | parallel | `--no-parallel` runs the fused serial core path |
| `--packed` | off | time the fused native file path with a packed `IO::Buffer` result (ignores `--limit-bytes`) |
| `--pretokenizer NAME` | none | required when TOKENIZER is a `.tiktoken` file; ignored otherwise |

### `gigatoken validate TOKENIZER FILES... [options]`

Confirms the native split-and-encode agrees with a Ruby-side split fed
through `encode_batch`. Takes `--doc-separator` and `--pretokenizer` as
above.

TOKENIZER is anything `Gigatoken::Tokenizer.load` accepts: a
`tokenizer.json` path or directory, a packaged encoding name, a Hub repo id,
or a `.tiktoken` file (with `--pretokenizer`). Errors print `error: ...` and
exit 1.

```sh
gigatoken bench cl100k_base owt_train.txt --doc-separator "<|endoftext|>" --packed
gigatoken bench lib/gigatoken/encodings/cl100k_base.tiktoken README.md --pretokenizer gpt4
gigatoken validate openai-community/gpt2 owt_train.txt --doc-separator "<|endoftext|>"
```
