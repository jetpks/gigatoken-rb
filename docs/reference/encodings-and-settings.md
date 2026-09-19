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

## Hub requests

Every request `Gigatoken::Hub` makes — the download behind
`Tokenizer.from_hub`, and behind `Tokenizer.load` for a repo id — carries a
10-second connect/read timeout, huggingface_hub's `HF_HUB_ETAG_TIMEOUT` /
`HF_HUB_DOWNLOAD_TIMEOUT` default. `Gigatoken::Hub.new(timeout: seconds)`
overrides it. Nothing partial is ever left in the cache: the body streams to
a temp file that is renamed into place once complete.

### Proxies

`http_proxy` / `HTTP_PROXY` and `https_proxy` / `HTTPS_PROXY` are honoured
per scheme (the lowercase spelling wins, as in `requests`), and suppressed
for a host named by `no_proxy` / `NO_PROXY` — a comma-separated list of host
suffixes, or `*` for every host. An http request travels to the proxy with
the absolute URI in its request line; an https one through a `CONNECT`
tunnel, then speaks TLS to the origin as if it had reached it directly. The
request phase carries the timeout either way; once the tunnel is up the body
read has no deadline of its own. A proxy that refuses the `CONNECT` is a
`Gigatoken::HubError` like any other transport failure.

The tunnel's connection is held by the proxy connection underneath it, so
both are released innermost-first once the body is written, and under the
same timeout: a connection a failure left mid-stream is abandoned rather than
waited on.

### Redirects

Up to 10 redirects are followed by hand, because two of the rules are the
Hub's own:

| Header | Rule |
| --- | --- |
| `Authorization` | Travels to a redirect on the same origin — huggingface.co answers a renamed repo with one — and is dropped crossing to another, which is how the LFS CDN (authenticating by signed URL) never sees the token. Same rule as huggingface_hub. |
| `x-repo-commit` | Taken from the first hop that carries it: an LFS file gets it on the `resolve/` hop, a renamed repo only on the hop that finally answers 200. |

### What is checked

| Value | Rule |
| --- | --- |
| `repo_id`, `filename`, `revision` | Not empty, no leading `/`, no NUL byte, no `.` or `..` path segment — checked before any request or cache access. |
| `revision` in the URL | Percent-encoded whole, like huggingface_hub's `quote(revision, safe="")`: `refs/pr/1` travels as `refs%2Fpr%2F1`. The repo id and the filename keep their slashes. |
| `x-repo-commit` response header | Must be a 40-character hex commit hash. It names the cache snapshot directory, so a server-chosen path is never trusted, and an endpoint that omits the header is not a Hub — its download could never be found in the cache again. |

Anything else is a `Gigatoken::HubError` naming the URL.

## Errors

### `Gigatoken::Error < StandardError`

The root of the hierarchy, and what to rescue to catch everything the gem
raises. A Rust panic never crosses the boundary as anything else. Raised
directly only for what none of the three subclasses below covers, such as a
CLI usage error.

### `Gigatoken::HubError < Error`

Everything `Gigatoken::Hub` raises, never a raw socket error: an HTTP error
status (immediately, body or no body), a refused connection, a proxy
refusing the `CONNECT` tunnel, a DNS failure, a timeout, a body cut short, a
malformed `HF_ENDPOINT`, and the repo id / revision / filename /
`x-repo-commit` checks. See [Hub requests](#hub-requests).

### `Gigatoken::InputError < Error`

A document the tokenizer cannot take: an untranscodable or invalid-byte
String, invalid UTF-8 on the SentencePiece path, an id outside the
vocabulary in `decode`.

### `Gigatoken::ModelError < Error`

A tokenizer that cannot be loaded: bad or hostile JSON, a missing file or
directory, an unknown or unpackable encoding name, a malformed `.tiktoken`.

`rescue Gigatoken::Error` still catches all four.

`ArgumentError`/`TypeError` come from Ruby argument handling as usual (a
non-String element in `encode_batch`, a bad `packed:` value).

## Related

- [`Gigatoken::Tokenizer`](tokenizer.md)
- [Load a tokenizer from any source](../how-to/load-a-tokenizer.md)
