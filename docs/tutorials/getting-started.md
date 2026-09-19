---
type: tutorial
---

# Getting started with `gigatoken`

**Goal:** install the gem, load a tokenizer that ships inside it, encode and
decode some text, and tokenize a batch of documents.

**Prerequisites:**

- Ruby 3.3 or newer
- On Apple Silicon macOS and x86_64/aarch64 Linux, nothing else: precompiled
  native gems are published. Elsewhere, a [Rust toolchain](https://rustup.rs/);
  `rust-toolchain.toml` pins the nightly the extension needs and `rustup`
  fetches it on the first build.

## 1. Install

```sh
bundle add gigatoken     # or: gem install gigatoken
```

## 2. Load a tokenizer

Four tiktoken encodings are vendored inside the gem, so this needs no
network and no cache directory:

```ruby
require "gigatoken"

tok = Gigatoken::Tokenizer.from_encoding("cl100k_base")
tok.vocab_size        # => 100277
tok.special_tokens    # => {"<|endoftext|>" => 100257, ...}
```

The same one-liner loads a `tokenizer.json`, a directory holding one, or a
HuggingFace Hub repo id — see
[Load a tokenizer from any source](../how-to/load-a-tokenizer.md).

## 3. Encode and decode

```ruby
ids = tok.encode("Hello, world!")   # => [9906, 11, 1917, 0]
tok.decode(ids)                     # => "Hello, world!"
```

`encode` returns one Array of Integer token ids. `decode` returns the bytes
as a binary (`ASCII-8BIT`) String; call `force_encoding("UTF-8")` if you
know the ids spell valid UTF-8.

## 4. Encode a batch

```ruby
docs = ["Hello, world!", "Another document."]
tok.encode_batch(docs)    # => [[9906, 11, 1917, 0], [14364, 2246, 13]]
```

`encode_batch` releases Ruby's GVL and encodes on the engine's worker pool,
so a batch of thousands of documents is where the throughput lives. For very
large batches, ask for a packed result instead of one Array per document:

```ruby
packed = tok.encode_batch(docs, packed: true)
packed.token_count    # => 7
packed[1]             # => [14364, 2246, 13]
```

See [Work with packed results](../how-to/use-packed-results.md).

## 5. Tokenize a file

```ruby
tok.encode_files("corpus.txt", separator: "<|endoftext|>")
# => one Array of ids per document between the separators
```

The file is read and split on the Rust side; the documents never become Ruby
Strings. See [Tokenize files without leaving Rust](../how-to/tokenize-files.md).

## Next steps

- [`Gigatoken::Tokenizer` reference](../reference/tokenizer.md)
- [Benchmarks](../explanation/benchmarks.md) — what to expect, and how the numbers were taken
