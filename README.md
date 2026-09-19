# gigatoken-rb

**12.4 GB/s / 2.8 billion tokens per second in Ruby.**

Zero-copy Ruby bindings for [marcelroed/gigatoken](https://github.com/marcelroed/gigatoken), the fastest open-source BPE tokenizer around.

| | Corpus | MB/s (median) | Gtok/s (median) |
|---|---|---|---|
| **gigatoken-rb** (this gem, Ruby) | 11.9 GB | **12,449** | **2.82** |
| gigatoken (Python wheel + [#38](https://github.com/marcelroed/gigatoken/issues/38)) | 11.9 GB | 12,226 | 2.77 |
| tiktoken (Python) | 1.35 GB | 69.7 | 0.0158 |
| tiktoken_ruby | 1.35 GB | 30.7 | 0.0070 |
| tokenizers gem (ankane) | 1.35 GB | 10.0 | 0.0023 |
| tokenizers (Python, Hugging Face) | 1.35 GB | 5.6 | 0.0013 |

Mac Studio M4 Max, OpenWebText, GPT-2 tokenizer; every library produces the same tokenization, gigatoken just does it faster. The Python row includes a fix for [marcelroed/gigatoken#38](https://github.com/marcelroed/gigatoken/issues/38) — a hidden memcpy in `shrink_to_fit()` we found while chasing the last of the Ruby–Python gap and sent upstream (one-line fix; without it the wheel lands around 7.4 GB/s).

**340x faster** than the fastest existing Ruby gem (tiktoken_ruby) and **1,050x faster** than the tokenizers gem. Full methodology & exact counts: [docs/explanation/benchmarks.md](docs/explanation/benchmarks.md).

## Install

```bash
gem install gigatoken
```

Precompiled native gems ship for Apple Silicon macOS (`arm64-darwin`) and x86_64/aarch64 Linux — on those platforms RubyGems grabs the binary automatically, no Rust toolchain, no compile wait. In a Bundler project it's one command:

```bash
bundle add gigatoken
```

(or drop `gem "gigatoken"` into the Gemfile yourself).

Anywhere else (or with `--platform ruby` to opt out of the binary), the extension builds from source. That needs a Rust toolchain: `rust-toolchain.toml` pins the nightly, and `rustup` fetches it automatically on first build.

## Use

```ruby
require "gigatoken"

tok = Gigatoken::Tokenizer.load("openai-community/gpt2")

tok.encode("Hello, world!")             # => [15496, 11, 995, 0]
tok.decode([15496, 11, 995, 0])         # => "Hello, world!"
tok.encode_batch(["Hello!", "Another"]) # => [[15496, 0], [6610]]

tok.vocab_size                          # => 50257
tok.special_tokens                      # => {"<|endoftext|>" => 50256}
```

`load` takes a `tokenizer.json` path, a directory holding one, a packaged tiktoken encoding name (`r50k_base`, `cl100k_base`, `o200k_base`), a HuggingFace Hub repo id, or a `.tiktoken` mergeable-ranks file, and dispatches on shape. Hub downloads run over socketry's `async-http` — no Python anywhere. Know what you have? Skip the dispatch:

```ruby
Gigatoken::Tokenizer.from_file("tokenizer.json")
Gigatoken::Tokenizer.from_hub("openai-community/gpt2", revision: "main")
Gigatoken::Tokenizer.from_tiktoken("cl100k_base.tiktoken", pretokenizer: "gpt4", special_tokens: {"<|endoftext|>" => 100257})
Gigatoken::Tokenizer.from_json(File.binread("tokenizer.json"))
```

A `.tiktoken` file holds mergeable ranks only — its pretokenization scheme and special tokens live in the code that defines the encoding, not the file — so `pretokenizer:` is a required keyword (one of `Gigatoken::Native.pretokenizer_names`: `gpt2`/`r50k`, `gpt4`/`cl100k`, `qwen2`, `qwen35`, `olmo3`, `deepseek_v3`, `o200k`, `nemotron`, `kimi`) and `special_tokens:` defaults to none. Nothing is guessed: an unknown scheme raises `Gigatoken::ModelError` naming the valid ones, and `Tokenizer.load` on a `.tiktoken` path with no `pretokenizer:` raises rather than silently picking one.

SentencePiece-BPE models (Llama, Gemma, Mistral — any `tokenizer.json` with `byte_fallback: true`) load through the same entry points and pick the right backend automatically. One difference: the SentencePiece core decodes text, so it validates input and raises `Gigatoken::InputError` on invalid UTF-8 instead of guessing.

### Packaged tiktoken encodings

`r50k_base`, `cl100k_base`, `o200k_base`, and `o200k_harmony` are vendored directly — mergeable ranks, pretokenizer scheme, and special-token table all shipped inside the gem (`lib/gigatoken/encodings/`; see `PROVENANCE.md` there for exact source URLs and hashes) — so all four resolve by name through both entry points entirely offline: no network access, no writable cache directory. `o200k_harmony` vendors no new file at all: it reuses `o200k_base.tiktoken`'s ranks and the `o200k` scheme verbatim, differing only in its special-token table (10 named control tokens — `<|start|>`, `<|message|>`, `<|end|>`, `<|return|>`, and so on — plus 1081 reserved slots; see `PROVENANCE.md` for the exact table). It's also the one packaged encoding not checked against `tiktoken_ruby`: that gem's 0.0.17 harmony table drops `<|endofprompt|>` where `openai/tiktoken` 0.9.0 keeps it at id 200018, so the oracle is the outlier here — `spec/gigatoken/differential_spec.rb` proves harmony instead by reduction to `o200k_base` plus a pinned special-token table.

```ruby
Gigatoken::Tokenizer.from_encoding("cl100k_base")
Gigatoken::Tokenizer.load("cl100k_base")          # same result — packaged names are
                                                  # checked before the Hub-repo-id shape
```

`p50k_base` and `p50k_edit` are deliberately not packaged: both load the same non-dense ranks (id 50256 is left free for `<|endoftext|>`), and the rank loader rejects non-dense ranks. Both entry points raise `Gigatoken::ModelError` explaining that, rather than `load` falling through to the Hub for a name that happens to look like a legacy repo id.

`encode` on a packaged tokenizer honours its special-token table: text containing `<|endoftext|>` (or any other literal special-token string) is tokenized as that special token, not as ordinary text. That matches [`tiktoken`](https://github.com/openai/tiktoken)'s `encode_with_special_tokens`, not its plain `encode`, which treats the same literal as ordinary text — a difference worth knowing if you're tokenizing untrusted input. To get tiktoken's non-honouring default instead, build a tokenizer from the same rank file with an empty special-token table:

```ruby
entry = Gigatoken::Encodings["cl100k_base"]
Gigatoken::Tokenizer.from_tiktoken(entry[:rank_file], pretokenizer: entry[:pretokenizer], special_tokens: {})
```

### Encode-cache budget

Each tokenizer's pretoken cache is capped process-globally (512 MiB per worker by default) so long-lived processes — a Rails worker, say — don't grow it unbounded; a full cache wipes back toward its seed level and refills, which costs a bit of re-computation but never changes encode output. Tune it before building tokenizers you want the new budget to apply to:

```ruby
Gigatoken.max_cache_bytes            # => 536870912 (512 MiB)
Gigatoken.max_cache_bytes = 64 << 20 # only tokenizers built after this see the new budget
Gigatoken.max_cache_bytes = nil      # unbounded

tok.cache_entries                    # => cached pretoken/unit count right now
```

### Tokenize files without leaving Rust

`encode_files` reads and tokenizes files entirely on the native side — document contents never materialize as Ruby objects. `.gz` and `.zst` decompress transparently.

```ruby
tok.encode_files("owt_train.txt", separator: "<|endoftext|>")

jsonl   = Gigatoken::Native::JsonlFileSource.new(["docs.jsonl"], field: "text")
parquet = Gigatoken::Native::ParquetFileSource.new(["docs.parquet"], column: "text")
tok.encode_files(jsonl)
```

### Packed results

Pass `packed: true` to `encode_batch` or `encode_files` and results land in a single `IO::Buffer` of u32 token ids instead of a ragged Array of Arrays — no per-token Ruby allocation, the fastest way out of the engine:

```ruby
packed = tok.encode_files("owt_train.txt", packed: true, separator: "<|endoftext|>")

packed.buffer       # => one IO::Buffer, every document's ids back to back
packed.lens         # => [12, 8, 41, ...] tokens per document
packed.token_count  # => total tokens
packed[3]           # => document 3's ids as an Array, on demand
```

### Errors

Everything the library raises is a `Gigatoken::Error`, never a raw Rust panic — and, under it, one of three: `Gigatoken::ModelError` when a tokenizer can't be loaded (bad or hostile JSON, a missing file or directory, an unknown or unpackable encoding name, a malformed `.tiktoken`), `Gigatoken::InputError` when a document or an id can't be taken (a String that won't transcode, invalid UTF-8 on the SentencePiece path, an id outside the vocabulary in `decode`), and `Gigatoken::HubError` for everything `Gigatoken::Hub` raises (HTTP status, transport, timeout, the repo-id and header checks). `rescue Gigatoken::Error` catches all three.

`encode` and `encode_batch` honour the String's encoding tag: UTF-8, US-ASCII and binary go through byte-wise, and a real non-UTF-8 encoding (ISO-8859-1, UTF-16LE, …) is transcoded first, so it gives the same ids as the same text read as UTF-8. See [the reference](docs/reference/tokenizer.md#input-encodings).

### Async

`encode_batch` and `encode_files` release the GVL for the whole encode; the parallelism runs on the engine's rayon pool, not Ruby threads. Under `Async`, give the fiber scheduler a worker pool (`ASYNC_SCHEDULER_WORKER_POOL=true`) and the calling fiber yields to the reactor too. Design notes: [docs/how-to/run-under-async.md](docs/how-to/run-under-async.md).

## CLI

```bash
gigatoken bench openai-community/gpt2 owt_train.txt --doc-separator "<|endoftext|>"
gigatoken validate openai-community/gpt2 owt_train.txt --doc-separator "<|endoftext|>"
```

`bench` reports MB/s and Mtok/s (`--packed` for the fused packed path, `--no-parallel` for the serial core). `validate` confirms native split-and-encode agrees with a Ruby-side split through `encode_batch`.

TOKENIZER also takes a bare `.tiktoken` file, which is where `--pretokenizer` comes in: the file carries mergeable ranks only, so the split regex has to come from the caller, same as `from_tiktoken` above. `--pretokenizer` takes one of the scheme names listed above for `pretokenizer:`:

```bash
gigatoken bench lib/gigatoken/encodings/cl100k_base.tiktoken README.md --pretokenizer gpt4
```

Leave it off against a `.tiktoken` TOKENIZER and both commands raise `Gigatoken::Error` naming the valid schemes instead of crashing; for every other TOKENIZER shape (`tokenizer.json`, a packaged name, a Hub repo id) `--pretokenizer` is accepted but ignored.

## Documentation

In-depth docs live under [`docs/`](docs/README.md), organized by
[Diátaxis](https://diataxis.fr/):

- **Tutorial:** [Getting started](docs/tutorials/getting-started.md)
- **How-to:** [Load a tokenizer](docs/how-to/load-a-tokenizer.md), [Tokenize files](docs/how-to/tokenize-files.md), [Packed results](docs/how-to/use-packed-results.md), [Cache budget](docs/how-to/tune-the-cache-budget.md), [Async](docs/how-to/run-under-async.md), [Measure](docs/how-to/measure-time-and-allocations.md)
- **Reference:** [`Tokenizer`](docs/reference/tokenizer.md), [`PackedResult`](docs/reference/packed-result.md), [File sources](docs/reference/file-sources.md), [Encodings and settings](docs/reference/encodings-and-settings.md), [CLI](docs/reference/cli.md)
- **Explanation:** [Benchmarks](docs/explanation/benchmarks.md), [Allocations](docs/explanation/allocations.md), [Async design](docs/explanation/async-design.md)

## Development

```bash
bundle install
bundle exec rake compile    # native extension (Rust nightly, via rust-toolchain.toml)
bundle exec rspec
bundle exec standardrb
ruby -Ilib bench/operations.rb   # every operation: i/s, objects and malloc per call
```

The Ruby layer is fiber-first throughout — no `Thread`, no `Mutex`; all parallelism lives in the core's rayon pool. CI runs ubuntu + macos × Ruby 3.3/3.4/4.0, and `release.yml` cross-builds the precompiled native gems (arm64-darwin, x86_64-linux, aarch64-linux).

## Fork status

This fork exists because I need fast tokenization in Ruby. The Rust core is changed as little as possible from upstream. Most of the python shell has been removed from this fork, but you can still find it [upstream](https://github.com/marcelroed/gigatoken).

SentencePiece works but — matching upstream — is less optimized than the BPE path.

Not ported/no current plans:
- the HF/tiktoken Python compat shims
- padded-batch matrices
- and BPE training

## Citation

The engine is Marcel Rød's gigatoken. If it shows up in your research, cite that:

```bibtex
@software{roed2026gigatoken,
  author = {Marcel R{\o}d},
  title = {{G}igatoken: SIMD and Cache Hierarchies for 1000x Faster Byte-Pair Encoding Tokenization on Modern CPUs},
  url = {https://github.com/marcelroed/gigatoken},
  year = {2026},
}
```

---

<details open>
<summary>AI Use Disclosure</summary>

The Rust engine is upstream's — see <a href="https://github.com/marcelroed/gigatoken#readme">upstream's AI-use disclosure</a> for how that was built (majority hand-crafted, AI-assisted toward the end).

The Ruby port in this fork is 100% AI generated using Fable 5 and Sonnet 5 via [space-architect](https://github.com/jetpks/space-architect) in ~24 hours.
</details>
