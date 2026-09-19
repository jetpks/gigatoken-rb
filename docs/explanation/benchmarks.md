# Benchmarks: methodology and full results

The README carries the headline numbers. This is everything behind them.

## Setup

Measured 2026-07-23 on a Mac Studio M4 Max (16 cores, 128 GB RAM, macOS 26)
against an OpenWebText reconstruction (`owt_train.txt`: 11,920,511,061 bytes,
2,393,319 docs, `<|endoftext|>`-separated — within 0.015% of upstream's own
recorded corpus), GPT-2 tokenizer (`r50k_base`, confirmed empirically
equivalent to `gpt2`).

Two payload sizes, because the comparison libraries run 100–2,000x slower than
gigatoken and a full-corpus pass would take them hours per run:

- The two gigatokens are measured on the full 11.9 GB — one warm process, a
  discarded warmup, then the median of three timed runs.
- The other four libraries are measured on a 1.35 GB slice of the same corpus —
  three fresh processes each, median.

Both gigatoken builds run with the mimalloc global allocator (this gem ships
it; the Python wheel was rebuilt with it) and `MIMALLOC_PURGE_DELAY=-1` — the
workaround for a macOS 26 allocator (xzm) crash on multi-GB frees that
otherwise kills either implementation above ~1.4 GB of input.

## Results

| Subject | Corpus | MB/s (median) | Gtok/s (median) | Notes |
|---|---|---|---|---|
| **gigatoken** (this gem, Ruby) | 11.9 GB | **12,278** | **2.78** | best 12,662; GVL released during the encode, the work runs on the engine's rayon pool. Batch API: `encode_batch`/`encode_files`, plus the packed `IO::Buffer` path. On the 1.35 GB slice: 10,519. |
| **gigatoken** (Python wheel — upstream anchor) | 11.9 GB | 7,400 | 1.68 | best 7,510; same rayon core underneath. Batch API: `encode_batch`. |
| tokenizers gem (ankane) | 1.35 GB | 10.0 | 0.0023 | `encode_batch_fast`, parallel across documents. |
| tiktoken_ruby | 1.35 GB | 30.7 | 0.0070 | single-threaded — the gem has no batch API, only a per-string `encode`. |
| tokenizers (Python, Hugging Face) | 1.35 GB | 5.6 | 0.0013 | `encode_batch_fast`, parallel across documents; repeats ranged 4.9–6.4 (upstream's own M4 Max table records 6.9). |
| tiktoken (Python) | 1.35 GB | 69.7 | 0.0158 | multi-threaded batch encode across documents. |

**Ruby-vs-Ruby, concretely:** on the same 1.35 GB slice this gem runs roughly
1,050x the tokenizers gem's throughput and roughly 340x tiktoken_ruby's.

## Update 2026-07-24: the gap was a hidden memcpy, and it's fixed

The 1.66x lead in the table above wasn't the Ruby seam being fast so much as
the Python path being robbed. Under mimalloc, the `flat.shrink_to_fit()` in
the core's `Committer::finish` reallocs — i.e. copies — the gathered token
buffer: 10.8 GB at full size, roughly half a second per pass. The gem's
packed path structurally avoids that trim (`finish_external` never had one);
the Python wheel always paid it. The fix is a one-line deletion, submitted
upstream as
[marcelroed/gigatoken#38](https://github.com/marcelroed/gigatoken/issues/38)
and already applied in this fork's core.

Rerun with the fix on both sides — same box, same corpus, same warm protocol:

| Subject | Total (median) | MB/s (median) | Gtok/s (median) |
|---|---|---|---|
| **gigatoken** (this gem, Ruby, packed) | 0.958 s | **12,449** | **2.82** |
| gigatoken (Python wheel + #38 fix) | 0.975 s | 12,226 | 2.77 |

Python's full-corpus `core_encode` dropped from 1.525 s to 0.962 s (best) —
the predicted memcpy, gone — and exact token parity (2,703,638,357) held
every run. These are the two gigatoken rows the README now carries: same
engine, same speed, within 2%. The unfixed wheel numbers remain in the table
above for the record.

## Token counts match

gigatoken — Ruby or Python, same engine — counts 2,703,638,357 tokens on the
full corpus and 306,287,417 on the slice, every run. The other four libraries
all count 306,017,245 on the slice: gigatoken's count minus exactly one
`<|endoftext|>` token per document boundary (gigatoken encodes the separators;
the others receive pre-split documents). Same underlying tokenization, wildly
different throughput.

## Caveats before trusting these numbers on your own workload

- The zero-copy input path only kicks in for documents whose bytes live in a
  heap allocation rather than being embedded in the Ruby object header itself.
  Under Ruby 4.0.6's Variable Width Allocation, that embed threshold falls
  somewhere between 512 B and 1 KB (512 B still embeds; 1024 B doesn't) —
  documents under roughly a kilobyte take a copy path instead.
- Every number above comes from one big file split into documents in-process —
  a whole-file, large-document shape. A workload made of many small documents,
  each already its own Ruby object, will land differently, for the reason
  above.

None of this repeats upstream's own core-engine numbers — those are measured
independently through the Python package and live at
[marcelroed/gigatoken#benchmarks](https://github.com/marcelroed/gigatoken#benchmarks).

## 0.2.1 thread-safety benchmark (2026-08-11)

Evidence for the single-`#encode` A/B claim in the [0.2.1] CHANGELOG entry:
that outlining `encode_contended` (`#[cold] #[inline(never)]`,
`ext/gigatoken/src/tokenizer.rs`) is what keeps the uncontended `#encode`
fast path free of an inlining-driven regression under this workspace's
`lto = "fat"`. Measured 2026-08-11 on an **Apple M2 Max, 12 cores, macOS
26.6.1**, Ruby 4.0.6 — a different machine from the M4 Max box the rest of
this file's numbers come from, and a shared dev machine with no attempt made
to quiesce other processes beyond closing other applications.

**Read "How much of this can you trust" at the end of this section before
quoting any figure from it.** The short answer: no size on this hardware
resolves the attribute's effect on its own — the same-build noise floor is
close enough to the real effect's size that it swallows it more often than
not. The instrument itself is honest (see below) and has power to detect
effects of a couple of percent or more; the real effect here, on this
machine, isn't reliably one of those.

Method: `ruby -Ilib bench/encode_ab.rb` (`GIGATOKEN_AB_ROUNDS=80`, the
default). Single `#encode` only (never `encode_batch`/`encode_files`) on
`cl100k_base`, at three sizes — short (45 B), medium (2,280 B), large
(228,000 B) — timed as the **median** of many calls per round (8,000
iterations/round at short, 40 at medium, 16 at large; all three sizes are
warmed once before any timed round). Each round measures arm A twice (A1,
A2) and arm B once, rotating which of the three runs first/second/third
across rounds so no arm is systematically advantaged by position. A1 is the
reported "A" and is what's compared against B; A2 feeds the A/A noise floor,
which is a bootstrap — the 98th percentile of many resampled median-deltas
drawn from the pooled A1+A2 samples — rather than a single point comparison,
so it can't land on a lucky exact tie. See `bench/encode_ab.rb`'s header for
the full design and why: this replaced a mean-of-per-round-samples statistic
that a single scheduler hiccup could move by double digits, and a floor that
was confounded with arm order and built from half the delta's sample count.

`#[cold]`/`#[inline(never)]` are compile-time attributes with no
Ruby-visible switch, so `bench/encode_ab.rb` itself can only measure
whatever build is currently installed — both its "A" and "B" arms below ran
the shipped (attributes-present) build. This is one representative run;
see "How much of this can you trust" for the full same-build evidence:

| Size | A | B | A/A noise floor | A/B delta |
|---|---|---|---|---|
| short | 0.33us | 0.33us | 2.40% | +0.68% (indistinguishable) |
| medium | 2.90us | 2.90us | 0.87% | -0.00% (indistinguishable) |
| large | 253.63us | 253.50us | 0.61% | -0.05% (indistinguishable) |

**Counterfactual (attributes removed):** ran three times each build,
manually — see `bench/encode_ab.rb`'s header comment for the exact
procedure (strip `#[cold]`/`#[inline(never)]` from `encode_contended` in
`ext/gigatoken/src/tokenizer.rs`, `bundle exec rake compile`, run the
harness, then revert and rebuild again). Comparing the median "A" reading
across each build's own runs:

| Size | Attributes present (shipped, 8 runs) | Attributes removed (3 runs) | Shipped A/A floor range | Delta |
|---|---|---|---|---|
| short | 0.32us | 0.32us | 2.23% – 2.89% | ~0% |
| medium | 2.88us | 2.90us | 0.87% – 2.65% | +0.69% |
| large | 250.88us | 252.00us | 0.57% – 1.10% | +0.45% |

Medium and large both move in the direction the outlining was added to
protect (attributes-removed reads slower), but at every size the
cross-build delta is at or below the *smallest* same-build floor observed
in either build's own runs — so by the harness's own standard, none of the
three sizes resolve this specific run of the effect. That's a smaller
cross-build delta at medium than an earlier measurement on this same
machine reported (~1%, back when the floor itself was miscalibrated); it is
still the same direction, just not clearing the more honest floor now in
place.

**Power check (synthetic, not committed):** to confirm the instrument can
resolve *something*, a throwaway copy of the harness inflated arm B's
recorded medium-size time by a known percentage before classification.
Across repeated runs: a 3% injected slowdown was caught every time (6/6,
labeled "slower"); 2% was also caught every time (6/6); 1% — the
neighborhood of the real effect above — was caught 3 of 8 times, the rest
reading "indistinguishable." The instrument has power, just not much margin
over the effect this specific attribute produces on this machine.

Reproduce with:

```
ruby -Ilib bench/encode_ab.rb
```

### How much of this can you trust

The instrument is honest: across 8 consecutive same-build runs (7 before
the counterfactual rebuild, 1 after restoring it), every reported `A/B
delta` at every size stayed under 2% and not one size was ever classified
`faster` or `slower` — the numbers above are one of those 8 runs, not a
cherry-picked one. The medium-size noise floor, the tightest AC gate,
stayed between 0.87% and 2.65% across those runs, comfortably under the 3%
ceiling.

What it can't do on this machine is resolve the ~0.5-1% effect the
attributes actually produce. The counterfactual table above shows why:
attributes-removed reads slower at medium and large, in the expected
direction, but the size of that difference is smaller than the noise floor
measured in several of the same-build control runs. The power check backs
this up directly — a *known* 1% injected effect was caught less than half
the time with this floor. Raising the floor's threshold would only inflate
the false-negative rate further without changing the underlying
measurement; the honest position is that on this hardware, at this
precision, no single reading from this instrument settles whether the
`#[cold]`/`#[inline(never)]` pair is worth its keep at any size — only its
*direction*, repeated across independent runs and now visible without the
false confidence of a two- to nine-percent phantom delta, is evidence. Keep
the attributes on the strength of that direction and the original
inline-regression measurement that motivated them, not on a number from
this harness.
