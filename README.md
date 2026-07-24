# Gigatoken

*Tokenize your text data at GB/s — from Ruby.*

Ruby bindings for [marcelroed/gigatoken](https://github.com/marcelroed/gigatoken), a SIMD BPE tokenizer core. This gem wraps it in an idiomatic, modern-Ruby API: fiber-friendly, no threads, and fast enough that counting tokens stops being something you estimate.

The Rust engine lives in-tree here and ships as a native gem (`magnus` + `rb_sys`); there's no Python involved. See [Fork status](#fork-status) for what changed on the way from upstream's Python package to this.

## Installation

```bash
gem install gigatoken
```
or in a Gemfile:
```ruby
gem "gigatoken"
```
There's no precompiled native gem yet, so installing builds the extension from source — you'll need a Rust toolchain for that. `rust-toolchain.toml` pins the nightly this gem is built against, and `rustup` fetches it automatically on first build.

## Loading a tokenizer

`Gigatoken::Tokenizer.load` figures out what you handed it — a `tokenizer.json` path, a directory holding one, a HuggingFace Hub repo id, or a `.tiktoken` mergeable-ranks file — and dispatches accordingly:
```ruby
require "gigatoken"

Gigatoken::Tokenizer.load("tokenizer.json")
Gigatoken::Tokenizer.load("path/to/model/dir")
Gigatoken::Tokenizer.load("openai-community/gpt2")
Gigatoken::Tokenizer.load("vocab.tiktoken")

Gigatoken::Tokenizer.load("openai-community/gpt2", revision: "main")
```
Hub downloads run over socketry's `async-http` — no Python, no `huggingface_hub` gem. If you already know the shape you have, skip the dispatch and call the constructor directly:
```ruby
Gigatoken::Tokenizer.from_file("tokenizer.json")        # a path, or a directory containing one
Gigatoken::Tokenizer.from_hub("openai-community/gpt2", revision: "main")
Gigatoken::Tokenizer.from_tiktoken("vocab.tiktoken")
Gigatoken::Tokenizer.from_json(File.binread("tokenizer.json"))
```

### SentencePiece models

Any `tokenizer.json` whose model has `byte_fallback: true` — Llama, Gemma, Mistral, and the rest of the SentencePiece-BPE family — is handled by the same four entry points above without any extra step: `Gigatoken::Native.load_hf_json` inspects that flag and picks a `Gigatoken::Native::SentencePieceTokenizer` or a `Gigatoken::Native::BPETokenizer`, and `Gigatoken::Tokenizer` wraps either one the same way.

The two backends do differ on one point: BPE is byte-level and never inspects the bytes it's given, but SentencePiece's encode core works on `&str`, so it has to. Every document going into `encode`/`encode_batch`, every file `encode_files` reads, and any `TextFileSource#separator` are checked for valid UTF-8 first, raising `Gigatoken::Error` if they aren't — a Ruby `String` tagged UTF-8 isn't guaranteed to actually be valid UTF-8, and this backend can't silently do the wrong thing with bytes it can't decode:
```ruby
tokenizer = Gigatoken::Tokenizer.from_file("sp_tokenizer.json") # byte_fallback: true

tokenizer.encode("hello world")                      # => [271, 276]

invalid = "hello \xFF".dup.force_encoding(Encoding::ASCII_8BIT)
tokenizer.encode(invalid)                             # raises Gigatoken::Error, "invalid UTF-8: ..."
```

## Encoding

The core operations: encode one string, encode a batch, decode ids back to bytes, and inspect the vocabulary.
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

### Reading straight from files

`encode_files` reads and tokenizes files entirely on the Rust side — the document contents never cross into Ruby as objects at all. Hand it a bare path (or array of paths) and it wraps them in a `TextFileSource` for you; JSONL and Parquet inputs need one of the other source classes instead:
```ruby
# A bare path, optionally split into documents on a separator
tokenizer.encode_files("owt_train.txt", separator: "<|endoftext|>")

# Or build the source explicitly
text = Gigatoken::Native::TextFileSource.new(["owt_train.txt"], separator: "<|endoftext|>")
jsonl = Gigatoken::Native::JsonlFileSource.new(["docs.jsonl"], field: "text")
parquet = Gigatoken::Native::ParquetFileSource.new(["docs.parquet"], column: "text")

tokenizer.encode_files(text)
tokenizer.encode_files(text, parallel: false) # run on the calling thread instead of the worker pool
```
`.gz` and `.zst` inputs decompress transparently, no extra flag needed.

### Packed results

Pass `packed: true` to `encode_batch` or `encode_files` and you get a `Gigatoken::PackedResult` back instead of a ragged Array of Arrays: a single `IO::Buffer` holding every document's token ids as u32 (native byte order), plus `lens` — the per-document token counts needed to slice it back apart. Skipping the ragged shape means skipping the per-token Ruby array allocation it costs, so this is the fastest way to get results out when you don't need plain Arrays:
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

### Async and the GVL

`encode_batch` and `encode_files` both release the GVL for the duration of the encode, so other Ruby fibers keep making progress while one is in flight:
```ruby
Async do
  tokenizer.encode_files(source)
end
```
That said, the *calling* fiber only yields to the reactor if the active `Fiber.scheduler` was built with a worker pool — set `ASYNC_SCHEDULER_WORKER_POOL=true` (one worker by default), or construct one explicitly. Without a worker pool, this call blocks the calling fiber exactly as it would outside `Async` — the GVL still comes free for other threads, just not for the reactor. See [docs/rb/async.md](docs/rb/async.md) for the full design writeup.

## CLI

```bash
gigatoken bench openai-community/gpt2 owt_train.txt --doc-separator "<|endoftext|>"
gigatoken validate openai-community/gpt2 owt_train.txt --doc-separator "<|endoftext|>"
```
`bench` times an encode and reports MB/s and Mtok/s — `--no-parallel` runs the serial core path instead of the worker pool, `--packed` benches the fused packed path. `validate` is a consistency check: it confirms `encode_files` (native split + encode) agrees with a Ruby-side split fed through `encode_batch`.

## Benchmarks

> **Every number below is a prediction, not a measurement.** They're projected from recorded intermediate benchmark runs on the reference box described below — three of the six subjects were measured at a smaller corpus size and scaled up (flagged per row). This table gets re-measured end to end and replaced with final values at release.

**Protocol being predicted:** the OpenWebText reconstruction corpus at full size, 11.9 GB (`owt_train.txt`: 11,920,511,061 bytes, 2,393,319 docs, `<|endoftext|>`-separated — within 0.015% of upstream's own recorded corpus size), tokenized with GPT-2's tokenizer (`r50k_base`, confirmed empirically equivalent to `gpt2`). One warm process per subject: a discarded warmup run followed by a median over at least three timed runs. Reference box: `studio.slush.systems`, a Mac Studio M4 Max, 16 cores, 128 GB RAM, macOS 26. Both gigatoken builds (Ruby and Python) run with the mimalloc global allocator and `MIMALLOC_PURGE_DELAY=-1` set — that env var works around a mimalloc purge stall seen on macOS 26 that's believed related to `xzm`; no link is given here, since writing this required no network access and nothing was available to verify a citation against.

| Subject | MB/s | Notes |
|---|---|---|
| **gigatoken** (this gem, Ruby) | ~12,300 *(predicted)* | GVL released during the encode; the actual work runs on the engine's rayon pool. Batch API: `encode_batch`/`encode_files`, plus the packed `IO::Buffer` path. Measured directly at this protocol: 12,280 MB/s median, 12,559 best. |
| **gigatoken** (Python wheel — upstream anchor) | ~7,500 *(predicted)* | Same rayon core underneath. Batch API: `encode_batch`. Measured directly at this protocol: 7,495 MB/s best. |
| tokenizers gem (ankane) | ~9–10 *(predicted)* | Single fresh process in the recorded run. Not yet measured at the 11.9 GB protocol — this is a scale-up from 9.6 MB/s recorded at 1.35 GB. |
| tiktoken_ruby | ~31 *(predicted)* | Single-threaded — the gem has no batch API, only a per-string `encode`. Measured directly at this protocol: 31.03 MB/s, consistent with 31.2 MB/s recorded at 1.35 GB. |
| tokenizers (Python, Hugging Face) | ~10 *(predicted)* | Single process in the recorded run, no batch API exercised. Not yet measured at the 11.9 GB protocol — scaled up from 10.1 MB/s recorded at 1.35 GB. |
| tiktoken (Python) | ~55–60 *(predicted)* | 16 threads, batch encode. Not yet measured at the 11.9 GB protocol — scaled up from 59.6 MB/s recorded at 1.35 GB. |

**Token counts match.** gigatoken — Ruby or Python, same engine — counts 2,703,638,357 tokens on the full corpus, every run. The other four libraries agree exactly with each other at 2,701,245,039 tokens: gigatoken's count minus 2,393,318 separator tokens. Same underlying tokenization, wildly different throughput.

**Ruby-vs-Ruby, concretely:** at the 1.35 GB scale this gem runs roughly 564x the tokenizers gem's throughput and roughly 175x tiktoken_ruby's.

Two things worth knowing before trusting these numbers on your own workload:
- The zero-copy input path only kicks in for documents whose bytes live in a heap allocation rather than being embedded in the Ruby object header itself. Under Ruby 4.0.6's Variable Width Allocation, that embed threshold falls somewhere between 512 B and 1 KB (512 B still embeds; 1024 B doesn't) — documents under roughly a kilobyte take a copy path instead.
- Every number above comes from one big file split into documents in-process — a whole-file, large-document shape. A workload made of many small documents, each already its own Ruby object, will land differently, for the reason above.

None of this repeats upstream's own core-engine numbers — those are measured independently through the Python package and live at [marcelroed/gigatoken#benchmarks](https://github.com/marcelroed/gigatoken#benchmarks).

## Development

```bash
bundle install
bundle exec rake compile    # builds the native extension (Rust nightly, via rust-toolchain.toml)
bundle exec rspec
bundle exec standardrb
```
Nothing in the Ruby layer touches `Thread`, `Mutex`, or `Monitor` — it's fiber-first throughout, and all the actual parallelism happens inside the core's rayon pool. CI (`ruby-ci.yml`) runs the suite across ubuntu + macos × Ruby 3.3/3.4/4.0; `ruby-gem.yml` cross-builds precompiled native gems (arm64-darwin, x86_64-linux, aarch64-linux) on demand.

## Fork status

This fork exists for one reason: to ship gigatoken as a Ruby gem. The Rust core itself is upstream's, touched as little as possible — a cargo feature gate plus some crate-root re-exports, so the `python` feature and its pyo3 bindings stay in the tree (unused, unshipped) purely to keep syncing from upstream painless. What's gone is the Python shell around it: its tests, its packaging, all removed. Want the Python package? That's [upstream](https://github.com/marcelroed/gigatoken).

What didn't come along: HF/tiktoken Python compatibility shims, padded-batch matrices, BPE training, and WordPiece (upstream doesn't have that last one either). SentencePiece tokenization works here, but — matching upstream — it's less optimized than the BPE path.

## Citation

The tokenizer engine itself is Marcel Rød's gigatoken. If it shows up in your research, cite that:

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

The Rust engine is upstream's — see <a href="https://github.com/marcelroed/gigatoken#readme">upstream's AI-use disclosure</a> for how that was built (majority hand-crafted, AI-assisted toward the end). The Ruby port in this fork is a different story: it was built AI-first, headless builder agents working iteration by iteration against human-and-AI-authored specifications with frozen acceptance criteria, every gate re-run cold at judging time, under human direction throughout.
</details>
