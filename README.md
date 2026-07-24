# gigatoken-rb

**12 GB/s / 2.8 billion tokens per second in Ruby.**

Ruby bindings for [marcelroed/gigatoken](https://github.com/marcelroed/gigatoken), the fastest open-source BPE tokenizer around — running **1.6x faster than upstream's own Python package**, on the same Rust engine.

| | Corpus | MB/s (median) | Gtok/s (median) |
|---|---|---|---|
| **gigatoken-rb** (this gem, Ruby) | 11.9 GB | **12,278** | **2.78** |
| gigatoken (Python wheel, upstream) | 11.9 GB | 7,400 | 1.68 |
| tiktoken (Python) | 1.35 GB | 69.7 | 0.0158 |
| tiktoken_ruby | 1.35 GB | 30.7 | 0.0070 |
| tokenizers gem (ankane) | 1.35 GB | 10.0 | 0.0023 |
| tokenizers (Python, Hugging Face) | 1.35 GB | 5.6 | 0.0013 |

Mac Studio M4 Max, OpenWebText, GPT-2 tokenizer; every library produces the same tokenization. **340x faster** than the fastest existing Ruby gem (tiktoken_ruby) and **1,050x faster** than the tokenizers gem.  Full methodology, exact counts, and the caveats that matter: [docs/rb/benchmarks.md](docs/rb/benchmarks.md).

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

`load` takes a `tokenizer.json` path, a directory holding one, a HuggingFace Hub repo id, or a `.tiktoken` mergeable-ranks file, and dispatches on shape. Hub downloads run over socketry's `async-http` — no Python anywhere. Know what you have? Skip the dispatch:

```ruby
Gigatoken::Tokenizer.from_file("tokenizer.json")
Gigatoken::Tokenizer.from_hub("openai-community/gpt2", revision: "main")
Gigatoken::Tokenizer.from_tiktoken("vocab.tiktoken")
Gigatoken::Tokenizer.from_json(File.binread("tokenizer.json"))
```

SentencePiece-BPE models (Llama, Gemma, Mistral — any `tokenizer.json` with `byte_fallback: true`) load through the same entry points and pick the right backend automatically. One difference: the SentencePiece core decodes text, so it validates input and raises `Gigatoken::Error` on invalid UTF-8 instead of guessing.

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

### Async

`encode_batch` and `encode_files` release the GVL for the whole encode; the parallelism runs on the engine's rayon pool, not Ruby threads. Under `Async`, give the fiber scheduler a worker pool (`ASYNC_SCHEDULER_WORKER_POOL=true`) and the calling fiber yields to the reactor too. Design notes: [docs/rb/async.md](docs/rb/async.md).

## CLI

```bash
gigatoken bench openai-community/gpt2 owt_train.txt --doc-separator "<|endoftext|>"
gigatoken validate openai-community/gpt2 owt_train.txt --doc-separator "<|endoftext|>"
```

`bench` reports MB/s and Mtok/s (`--packed` for the fused packed path, `--no-parallel` for the serial core). `validate` confirms native split-and-encode agrees with a Ruby-side split through `encode_batch`.

## Development

```bash
bundle install
bundle exec rake compile    # native extension (Rust nightly, via rust-toolchain.toml)
bundle exec rspec
bundle exec standardrb
```

The Ruby layer is fiber-first throughout — no `Thread`, no `Mutex`; all parallelism lives in the core's rayon pool. CI runs ubuntu + macos × Ruby 3.3/3.4/4.0, and `release.yml` cross-builds the precompiled native gems (arm64-darwin, x86_64-linux, aarch64-linux).

## Fork status

This fork exists because I need fast tokenization in Ruby. The Rust core is changed as little as possible from upstream. Most of the python shell has been removed from this fork, but you can still find it [upstream](https://github.com/marcelroed/gigatoken).

Not ported/no current plans:
- the HF/tiktoken Python compat shims
- padded-batch matrices
- and BPE training

SentencePiece works but — matching upstream — is less optimized than the BPE path.

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

The Ruby port in this fork is 100% AI generated using Fable 5 and Sonnet 5 via [space-architect](https://github.com/jetpks/space-architect) over ~24 hours.
</details>
