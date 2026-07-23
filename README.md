# Gigatoken

*Tokenize your text data at GB/s — from Ruby.*

A Ruby gem binding [marcelroed/gigatoken](https://github.com/marcelroed/gigatoken)'s SIMD tokenizer core: the fastest BPE tokenizer for language modeling, exposed through an idiomatic modern-Ruby API. Token counting at these speeds is effectively free — no more estimating.

This fork carries the same Rust engine in-tree and ships it as a native gem (`magnus` + `rb_sys`). The upstream Python shell is still present but is not this fork's target — see [Fork status](#fork-status).

## Installation

```bash
gem install gigatoken
```
or in your Gemfile:
```ruby
gem "gigatoken"
```
No precompiled native gem is published yet, so this builds the extension from source. That needs a Rust toolchain — `rust-toolchain.toml` pins the nightly it's built against, and `rustup` installs that toolchain automatically the first time you build.

## Loading

`Gigatoken::Tokenizer.load` accepts any of four source shapes — a `tokenizer.json` path, a directory containing one, a HuggingFace Hub repo id, or a `.tiktoken` mergeable-ranks file — and dispatches to the right constructor:
```ruby
require "gigatoken"

Gigatoken::Tokenizer.load("tokenizer.json")
Gigatoken::Tokenizer.load("path/to/model/dir")
Gigatoken::Tokenizer.load("openai-community/gpt2")
Gigatoken::Tokenizer.load("vocab.tiktoken")

Gigatoken::Tokenizer.load("openai-community/gpt2", revision: "main")
```
Hub fetches go through socketry `async-http` — no Python, no `huggingface_hub`. Each shape also has an explicit constructor, if you already know which one you have:
```ruby
Gigatoken::Tokenizer.from_file("tokenizer.json")        # path, or a directory containing one
Gigatoken::Tokenizer.from_hub("openai-community/gpt2", revision: "main")
Gigatoken::Tokenizer.from_tiktoken("vocab.tiktoken")
Gigatoken::Tokenizer.from_json(File.binread("tokenizer.json"))
```

### SentencePiece

A `tokenizer.json` whose model has `byte_fallback: true` (Llama, Gemma, Mistral, and other SentencePiece-BPE families) loads automatically through the same `.load`/`.from_file`/`.from_json` entry points above — `Gigatoken::Native.load_hf_json` picks a `Gigatoken::Native::SentencePieceTokenizer` or a `Gigatoken::Native::BPETokenizer` based on that flag, and `Gigatoken::Tokenizer` wraps either one transparently. No separate API to learn.

One contract is stricter for the SentencePiece backend than for BPE: BPE is byte-level and trusts raw bytes as-is, but SentencePiece's encode cores operate on `&str`, so every document passed to `encode`/`encode_batch`, every file's contents read by `encode_files`, and a `TextFileSource`'s `separator:` are validated as UTF-8 and raise `Gigatoken::Error` on invalid input (a Ruby `String` carries no UTF-8 guarantee, even when tagged as one).

## Core API

```ruby
tokenizer = Gigatoken::Tokenizer.load("openai-community/gpt2")

tokenizer.encode("Hello, world!")                    # => [15496, 11, 995, 0]
tokenizer.encode_batch(["Hello, world!", "Another one"])
tokenizer.decode([15496, 11, 995, 0])                 # => "Hello, world!"

tokenizer.vocab_size                                  # => 50257
tokenizer.vocab                                       # => {0 => "!", 1 => "\"", ...}
tokenizer.merges                                      # => [[" ", "t"], [" ", "a"], ...]
tokenizer.special_tokens                              # => {"<|endoftext|>" => 50256}
```

### Encoding files

`encode_files` tokenizes whole files in Rust, without the documents ever becoming Ruby objects. Bare paths are wrapped in a `TextFileSource` automatically; JSONL and Parquet need one of the other Native source classes:
```ruby
# Bare path(s), optionally split into documents on a separator
tokenizer.encode_files("owt_train.txt", separator: "<|endoftext|>")

# Or an explicit Native source
text = Gigatoken::Native::TextFileSource.new(["owt_train.txt"], separator: "<|endoftext|>")
jsonl = Gigatoken::Native::JsonlFileSource.new(["docs.jsonl"], field: "text")
parquet = Gigatoken::Native::ParquetFileSource.new(["docs.parquet"], column: "text")

tokenizer.encode_files(text)
tokenizer.encode_files(text, parallel: false) # encode on the calling thread instead of the worker pool
```
`.gz` and `.zst` files decompress transparently.

### Packed results

`encode_batch`/`encode_files` accept `packed: true` to return a `Gigatoken::PackedResult` instead of a ragged Array of Arrays: one `IO::Buffer` of every document's token ids (u32, native byte order) plus `lens`, each document's token count. This skips the per-token Ruby array materialization the ragged shape costs, and is the fastest way to consume results when you don't need plain Arrays:
```ruby
packed = tokenizer.encode_files(text, packed: true)

packed.buffer                                         # => an IO::Buffer
packed.lens                                            # => [12, 8, 41, ...] (tokens per document)
packed.size                                             # => number of documents
packed.token_count                                      # => total tokens across every document
packed[3]                                                # => Array of token ids for document 3, materialized on demand
packed.each { |ids| ... }                                # each document's token ids, in order
packed.to_a                                              # => the same ragged shape packed: false returns
```

### Async

`encode_batch`/`encode_files` release the GVL, so wrapping them in `Async` lets other fibers keep running while an encode is in flight:
```ruby
Async do
  tokenizer.encode_files(source)
end
```
The calling fiber itself only yields to the reactor when the active `Fiber.scheduler` was built with a worker pool (`ASYNC_SCHEDULER_WORKER_POOL=true`, one worker by default) — without one, this blocks exactly as it would outside `Async`. See [docs/rb/async.md](docs/rb/async.md) for the full design note.

## CLI

```bash
gigatoken bench openai-community/gpt2 owt_train.txt --doc-separator "<|endoftext|>"
gigatoken validate openai-community/gpt2 owt_train.txt --doc-separator "<|endoftext|>"
```
`bench` reports MB/s and Mtok/s (`--no-parallel` for the serial core path, `--packed` to bench the packed fused path); `validate` cross-checks `encode_files` against a Ruby-side split plus `encode_batch`.

## Performance

The engine is unchanged from upstream — [its benchmarks](https://github.com/marcelroed/gigatoken#benchmarks) (measured through the Python package: up to tens of GB/s, roughly 10–1000× HuggingFace tokenizers depending on tokenizer family and CPU) characterize the core this gem binds. Ruby-side throughput depends on the seam: the packed `encode_files` path stays within ~10% of the Python CLI on the corpora we've measured, with token counts matching exactly. No Ruby-side multipliers are claimed here — run `gigatoken bench` against your own tokenizer and files to measure it on your hardware.

## Development

```bash
bundle install
bundle exec rake compile    # builds the native extension (Rust nightly, via rust-toolchain.toml)
bundle exec rspec
bundle exec standardrb
```
The Ruby layer is fiber-first: no `Thread`, `Mutex`, or `Monitor` — parallelism lives in the core's rayon pool. CI runs the suite on ubuntu + macos × Ruby 3.3/3.4/4.0 (`ruby-ci.yml`); `ruby-gem.yml` cross-builds precompiled native gems (arm64-darwin, x86_64-linux, aarch64-linux) on demand.

## Fork status

This fork exists to ship the Ruby gem. The Rust core is upstream's, kept minimally patched (a cargo feature gate and crate-root re-exports); the Python shell (`gigatoken/`, `pyproject.toml`) is currently retained byte-identical but unmaintained here, and will be removed eventually — use [upstream](https://github.com/marcelroed/gigatoken) for the Python package.

Not ported: HF/tiktoken Python compat shims, padded-batch matrices, BPE training, WordPiece (upstream lacks it too). SentencePiece tokenization works but is less optimized than BPE, matching upstream.

## Citation

The tokenizer engine is Marcel Rød's gigatoken. If you use it in your research, please cite:

```bibtex
@software{roed2026gigatoken,
  author = {Marcel R{\o}d},
  title = {{G}igatoken: SIMD and Cache Hierarchies for 1000x Faster Byte-Pair Encoding Tokenization on Modern CPUs},
  url = {https://github.com/marcelroed/gigatoken},
  year = {2026},
}
```

---

<details>
<summary>AI Use Disclosure</summary>

The Rust engine is upstream's; see <a href="https://github.com/marcelroed/gigatoken#readme">upstream's AI-use disclosure</a> for how it was built (majority hand-crafted, AI-assisted in the final stages). The Ruby port in this fork was built AI-first: headless builder agents working iteration-by-iteration against human-and-AI-authored specifications with frozen acceptance criteria, every gate re-run cold at judging, under human direction throughout.
</details>
