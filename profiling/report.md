# Single-threaded cold-encode CPU profile — gigatoken encode_st, 10 GB OWT

Machine: Apple M4 Max (12P+4E), macOS 27, corpus `~/data/owt_train.txt` in page
cache. Workload: one cold `ENCODE_MB=10000` pass of `benches/encode_st.rs` with
`data/gpt2_tokenizer.json` (2,279,617,884 tokens, ~2.08 G pretokens).
Baseline bench (no profiler attached): **10.48–10.74 s, ~950 MB/s**.

## 1. Setup

### Perf parity
`[profile.bench]` in `Cargo.toml` now inherits `release` (fat LTO, identical
codegen) and adds `debug = true`, `strip = "none"`,
`split-debuginfo = "packed"` (emits a `.dSYM`). Debug info does not change
machine code. Verified sequentially on identical full-pass runs:

| binary | wall (encode) | tokens |
|---|---|---|
| plain release bench (pre-change) | 10.48 s | 2,279,617,884 |
| bench + debuginfo | 10.65 s | 2,279,617,884 |
| bench + debuginfo under samply 4 kHz | 10.74 s / 11.38 s (2 runs) | 2,279,617,884 |

1.6 % delta plain-to-debuginfo, within this bench's +-8 % run-to-run band.
The old `[profile.profiling]` (`lto = false`) was NOT used — it changes inlining.

### Tools chosen
- **Primary: samply 0.13.1** (`record --save-only -r 4000 --main-thread-only
  --unstable-presymbolicate`). Firefox-profiler JSON; ~48 k samples per full
  pass; <1–7 % overhead. `--save-only` output is **unsymbolicated**, so
  `analyze.py` resolves every unique (lib, address) itself:
  - main binary: `atos -o <dSYM> -arch arm64 -offset -i -fullPath`, which
    expands **inline frames with exact source file:line at each sampled
    address** — this is what makes attribution trustworthy under fat LTO +
    `#[inline(always)]`. atos output groups are delimited by interleaving a
    sentinel (unmapped) address between queries; blank-line splitting is
    unreliable.
  - system dylibs (dyld shared cache, unreadable by atos): symbol ranges from
    samply's `--unstable-presymbolicate` sidecar (`*.syms.json`).
  - Rust v0 names demangled via `rustfilt`, generics stripped for readability.
- **PMU: `xctrace record --template 'CPU Counters'`** (Instruments 16, "CPU
  Bottlenecks" guided mode). Exported `MetricTable` / `RemarksByThread` /
  `CountingModeSamples` tables via `xctrace export --xpath`, resolving the
  XML's id/ref compression and weighting rows by their duration/weight
  (unweighted counting over-counts rare rows).
- Rejected: xctrace Time Profiler as primary (no per-address inline expansion
  in exported XML); `sample` (no weights, poor inline support).

### Sanity checks
- Profile total vs wall: samply attributes 10.69 s to the encode phase vs
  10.74 s measured by the bench (99.5 %). (The read phase under-counts in
  sample weights — kernel copyin samples get coalesced — but
  `threadCPUDelta` recovers its 1.6 s CPU.)
- Reproducibility, two independent 10 GB traces (A vs B), % of total process
  samples: walker 45.8/45.4, merge 15.1/14.2, driver 13.5/13.4, cache probe
  12.6/13.1, key-pack 5.6/5.2 — all major buckets agree within ~1 pt.
- Top functions match the code's expectations (walker loop, cache probe,
  driver loop, miss-path merge).

## 2. Where the 10.5 s goes

Buckets are weighted self time with inline frames resolved, categorized by
walking each sample's (inline-expanded) stack outward until a rule matches.
From `traces/samply_10gb_b.json.gz` (encode phase = 10.69 s of 11.62 s total
process time; % below are of the encode phase):

| bucket | % of encode | seconds | ns/pretoken |
|---|---|---|---|
| pretokenizer walker (`src/pretokenize/fast/*`, span+mask loops) | 49.4 % | 5.28 s | 2.54 |
| BPE merge, cache-miss fallback (`bpe_merge_symbols_small`, `encode_pretoken_miss`, hashbrown probes) | 15.4 % | 1.65 s | 0.79 |
| encode driver (`memoized_encode` probe loop, output callback) | 14.6 % | 1.56 s | 0.75 |
| pretoken cache probe/insert (`ShortPretokenCache::get/insert/prefetch`) | 14.2 % | 1.52 s | 0.73 |
| cache key pack (`pack_pretoken_key`) | 5.6 % | 0.60 s | 0.29 |
| memcpy/memset, malloc/free, syscalls, other | ~0.9 % | ~0.1 s | — |

Off-CPU during encode: wall 10.69 s vs thread-CPU 10.23 s -> ~0.4 s (~4 %)
off-CPU/faults; the table is touched cold each pass but page-fault cost is
minor at this scale. Outside encode: corpus read+UTF-8 validation ~1.6 s CPU
(`String::from_utf8` — `run_utf8_validation` is 7 % of total process samples,
all under `load_owt_input`), tokenizer load ~0.04 s.

Top self-time functions (% of total process, trace B):

```
17.7%  fill_spans_keyed_with            [pretokenize/mod.rs]   (per-pretoken pull+hash+store loop)
11.0%  MaskState::next_span             [fast/mask.rs]
10.2%  ShortPretokenCache::get          [bpe/pretoken_cache.rs]
 9.1%  Tokenizer::memoized_encode       [bpe/tiktoken.rs]      (probe/emit loop)
 7.1%  run_utf8_validation              [core]                 <- load_owt_input (read phase)
 5.1%  pack_pretoken_key                [pretokenize/mod.rs]
 4.4%  r50k::batch_masks                [fast/mask.rs]         (NEON classify)
 4.1%  encode_st::main::{closure#3}     [bench]                (per-pretoken output callback)
 3.7%  bpe_merge_symbols_small          [bpe/mod.rs]
 3.1%  neon::vandq_u8                   <- mask::movemask64
 2.1%  r50k::ascii_batch_algebra        [fast/r50k.rs]
 2.0%  ShortPretokenCache::prefetch
```

Hot lines (samply + atos line attribution, % of total process):

```
 8.96%  pretokenize/mod.rs:194   batch.hashes[n] = h;        (hash math + stores of the pull loop land here)
 4.85%  bpe/tiktoken.rs:378      let (key, h) = (batch.keys[i], batch.hashes[i]);
 4.74%  bpe/pretoken_cache.rs:156  if e.key == key {          (probe load stall)
 3.01%  pretokenize/mod.rs:192   batch.spans[n] = span;
 2.67%  bpe/tiktoken.rs:383      if val & VAL_SPILL == 0 {    (unpredictable spill branch)
 2.58%  pretoken_cache.rs:159    if e.key == EMPTY_KEY {
 2.43%  pretokenize/mod.rs:97    let mask = PACK_MASK[n];     (dependent table load in key pack)
 1.54%  fast/mask.rs:577         if self.rem != 0 {           (span-boundary branch)
```

## 3. Microarchitectural findings (xctrace CPU Counters, encode window, P-core)

4.0 GHz sustained, 41.8 G cycles over the 10.4 s encode window. Instruments'
top-level bandwidth decomposition (duration-weighted means):

| component | fraction |
|---|---|
| Useful (retiring) | **54.9 %** |
| **Discarded (bad speculation / mispredicted paths)** | **25.1 %** |
| Instruction Processing bottleneck (backend: data deps / memory latency) | 11.4 % |
| Instruction Delivery bottleneck (frontend) | 7.4 % |

Instruments remarks: 151x "High Discarded: incorrect speculative execution is
wasting bandwidth", 26x "High Processing: serial data dependences with
possibly long memory latencies", 49x "High Delivery" (nearly all in the read
phase's UTF-8 validation, 90.8 % of delivery-bottleneck weight).

Discarded-bandwidth weight by function (`CountingModeSamples`, weighted):
`fill_spans_keyed_with` 17.7 %, `MaskState::next_span` 10.5 %,
`ShortPretokenCache::get` 9.7 %, `memoized_encode` 9.3 %,
`run_utf8_validation` 7.4 %, `pack_pretoken_key` 5.3 %, `batch_masks` 4.7 %,
bench output closure 4.6 %.

**The single biggest inefficiency in cold encode is branch misprediction, not
memory bandwidth**: a quarter of P-core issue bandwidth is thrown away, and it
concentrates in the per-pretoken span-extraction loop and the probe/emit
loop's data-dependent branches (`rem != 0` bit-walk, span-length paths in
`pack_pretoken_key`, hit/spill/miss branch triad, probe hit-vs-empty).
Memory latency is second-order (11 %) and concentrated where expected: the
first-touch probe load (`e.key == key`) — the chunked prefetch is doing its
job but not perfectly.

## 4. Top 5 optimization targets the data supports

1. **De-branch the per-pretoken pull loop** (`fill_spans_keyed_with` +
   `MaskState::next_span`, 28 % of encode self time and the largest share of
   the 25 % discarded bandwidth). The `rem != 0` / scalar-vs-mask /
   segment-refill branch ladder is taken per pretoken with data-dependent
   direction. Candidates: extract all span boundaries of a 64-byte block
   branchlessly into a small buffer before the per-span work; unroll the
   bit-walk on `rem` with `count_ones()` iterations instead of a
   `while rem != 0` exit test.
2. **Shrink mispredicts in the probe/emit triad** (`memoized_encode` 14.6 % +
   probe 14.2 %). The `key != 0` / hit-vs-miss / `VAL_SPILL` branches are
   per-pretoken; the spill branch alone burns 2.7 % of the process. With ~90 %
   1-token and ~98 % <=2-token values, emitting the 2-token inline pair
   unconditionally into a local buffer and branchlessly advancing by `len`
   (deferring the rare arena-spill path) would remove the hottest
   unpredictable branch pair.
3. **Cut the probe's dependent-load stall** (`pretoken_cache.rs:156`, 4.7 % of
   process, and the dominant "Processing bottleneck" source). Ideas the data
   supports: issue the *second* probe slot's prefetch too (collisions walk
   `idx+1`), or pack key-tags so the common hit compares 8 bytes before
   touching the full 16-byte key; measure prefetch distance (currently one
   chunk ~ 256 pretokens ahead — CPU-Counters shows residual latency).
4. **Cheapen `pack_pretoken_key`** (5.6 % of encode + 5.3 % of discarded
   weight). The `PACK_MASK[n]` table load (2.4 % of process) is a dependent
   load on the critical path and the page-boundary branch is per-span:
   compute the mask arithmetically (shift-based, branchless for n<16) and
   consider unconditional 16-byte loads with a padded input buffer to delete
   the boundary branch entirely.
5. **Speed the cold-miss fallback** (15.4 % of encode on a cold pass;
   disappears warm). `bpe_merge_symbols_small` plus hashbrown pair-rank
   probes (`ProbeSeq::move_next` + `equivalent_key` ~ 2.5 %) dominate;
   `core::intrinsics::likely <- bpe_merge_symbols_small` (2.2 %) marks the
   merge-scan branch. A flatter rank lookup (e.g. perfect-hash or sorted-pair
   array for the small-symbol case) attacks the multi-thread cold-throughput
   end goal directly since every worker pays this until its cache warms.

Not encode, but a free end-to-end win: the input `String::from_utf8`
validation is 0.8–1.6 s of every run (`load_owt_input`); feeding the
encoder unvalidated `&[u8]` (it already handles bytes) removes it.

## 5. Artifacts

Scripts (this directory, committed on branch `encode-opt-main`; working copies
in `/private/tmp/claude-501/-Users-marcel-git-tokers/a5da8d2f-2bd4-46cc-a1b4-c98fb115a1cc/scratchpad/profiling/`):
- `profile.sh [MB] [samply|counters|both] [label]` — builds the parity bench
  binary, records a cold-run trace (one process at a time).
- `analyze.sh <trace>` / `analyze.py` — samply JSON -> `top_functions.txt`
  (phases, buckets, top-30 inline-resolved), `collapsed.folded` (flamegraph
  format, inline frames expanded), `hot_lines.txt` (source-line annotation).
- `pmu_summary.py` — xctrace CPU Counters trace -> bottleneck fractions,
  cycle rates, remarks (handles XML ref-compression + duration weighting).

Raw traces (kept on disk, not committed), under
`/private/tmp/claude-501/-Users-marcel-git-tokers/a5da8d2f-2bd4-46cc-a1b4-c98fb115a1cc/scratchpad/profiling/traces/`:
- `samply_10gb_a.json.gz`, `samply_10gb_b.json.gz` (+ `.syms.json` sidecar,
  `.bin` binary pointer) — full 10 GB cold passes, 4 kHz; analyses in
  `samply_10gb_{a,b}_analysis/`.
- `counters_10gb.trace` — xctrace CPU Counters, full 10 GB pass (+ exported
  `*.MetricTable.xml`, `*.RemarksByThread.xml`, `*.CountingModeSamples.xml`).
- `counters_1gb.trace`, `samply_300mb_test.json.gz`, `samply_smoke.json.gz` —
  pipeline validation runs.
- Profiled binary + dSYM:
  `scratchpad/opt/target/release/deps/encode_st-34b2b318116cca99{,.dSYM}`
  (plain-release parity binary preserved as
  `scratchpad/profiling/encode_st_plain`).
