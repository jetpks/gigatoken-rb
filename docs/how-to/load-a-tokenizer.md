---
type: how-to
---

# Load a tokenizer from any source

`Gigatoken::Tokenizer.load` takes one argument and dispatches on its shape.
When you already know what you have, call the specific constructor and skip
the dispatch.

## A packaged tiktoken encoding, by name

```ruby
Gigatoken::Tokenizer.from_encoding("cl100k_base")
Gigatoken::Tokenizer.load("cl100k_base")          # same result
```

`r50k_base`, `cl100k_base`, `o200k_base` and `o200k_harmony` ship inside the
gem (ranks, pretokenizer scheme and special-token table), so they load
offline. Packaged names are checked before the Hub-repo-id shape, so a bare
name like `o200k_base` never reaches the network. `p50k_base` and
`p50k_edit` are known but deliberately not packaged; asking for either
raises `Gigatoken::Error` saying why.

## A `tokenizer.json` file, or a directory containing one

```ruby
Gigatoken::Tokenizer.from_file("path/to/tokenizer.json")
Gigatoken::Tokenizer.from_file("path/to/model-dir/")
Gigatoken::Tokenizer.from_json(File.binread("tokenizer.json"))   # already in memory
```

SentencePiece-BPE models (`byte_fallback: true` — Llama, Gemma, Mistral)
load through the same calls and pick the SentencePiece backend automatically.

## A `.tiktoken` mergeable-ranks file

A `.tiktoken` file holds ranks only. Its pretokenization scheme and
special tokens live in the code that defines the encoding, so you must name
the scheme, and may pass the special-token table:

```ruby
Gigatoken::Tokenizer.from_tiktoken(
  "cl100k_base.tiktoken",
  pretokenizer: "gpt4",
  special_tokens: {"<|endoftext|>" => 100257}
)
```

Scheme names are `Gigatoken::Native.pretokenizer_names`: `gpt2`/`r50k`,
`gpt4`/`cl100k`, `o200k`, `qwen2`, `qwen35`, `olmo3`, `deepseek_v3`,
`nemotron`, `kimi`. `load` on a `.tiktoken` path without `pretokenizer:`
raises rather than guessing.

## A HuggingFace Hub repo

```ruby
Gigatoken::Tokenizer.from_hub("openai-community/gpt2")
Gigatoken::Tokenizer.from_hub("openai-community/gpt2", revision: "main")
Gigatoken::Tokenizer.load("openai-community/gpt2")   # dispatches here for org/name shapes
```

The download goes over `async-http`, honours `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`,
or the token file `hf auth login` writes), talks to `HF_ENDPOINT` when that is
set (a mirror, or a local server) and huggingface.co otherwise, and lands in
the standard HuggingFace cache (`HF_HUB_CACHE`, then `$HF_HOME/hub`), so later
loads by this gem or by `huggingface_hub` are served from disk.

Requests time out after 10 seconds — huggingface_hub's default — and follow
`http_proxy` / `https_proxy` (with `no_proxy`) when the environment sets them.
Every failure is a `Gigatoken::HubError` (a `Gigatoken::Error`) naming the URL,
including a refused connection, a proxy refusing the `CONNECT` tunnel, a DNS
failure and the timeout itself.

A revision may be nested (`revision: "refs/pr/1"`), and is percent-encoded in
the URL like huggingface_hub does it. The repo id, the filename and the
revision may not contain a `.` or `..` path segment, a leading `/` or a NUL
byte, and the response's `x-repo-commit` header must be a commit hash: those
name directories in your cache, and the header comes from the server.

To point at a mirror, or a local server in tests, inject the client:

```ruby
hub = Gigatoken::Hub.new(endpoint: "http://localhost:8080", timeout: 30)
Gigatoken::Tokenizer.load("org/model", hub: hub)
```

An endpoint that does not send `x-repo-commit` is not a Hub — a plain static
mirror will be refused rather than downloaded into a snapshot the cache could
never find again.

`load` builds a `Gigatoken::Hub` only when the source turns out to be a
repo id; packaged encodings and local files never construct one.

## Related

- [`Gigatoken::Tokenizer` reference](../reference/tokenizer.md)
- [Encodings, settings and errors](../reference/encodings-and-settings.md)
