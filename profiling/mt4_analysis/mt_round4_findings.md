# MT encode profile analysis — samply_mt_round4 (10 GB cold, GPT-2, 16 threads, M4 Max)

Trace: `samply_mt_round4.json.gz` of the round-3 mainline (encode-opt-main
@ab65161: fused copy+drop gather, atomic in-order LPT handout). Run measured
1.336 s encode = 7487 MB/s. Analysis: `analyze_mt.py` (round-3 tool,
unmodified), outputs in this directory. All CPU accounting uses
`threadCPUDelta` (kernel user+system µs).

## The 1336 ms window (t = 1963.1 → 3298.8), bucket by bucket

| phase | wall ms | % | CPU ms | notes |
|---|---|---|---|---|
| worker fork + seed | ~32 | 2.4% | 289 memset + 84 seed | 16 × 128 MB table memset (`alloc_zeroed`, 2 MiB align → explicit `__bzero`), bandwidth-bound ~65 GB/s aggregate |
| encode steady state | ~1090 | 81.6% | 14,669 | 13.45 / 16 threads busy avg; per-worker encode CPU 846–972 ms |
| straggler tail | (inside encode) | 0.5% | — | last-encode spread 3061.5→3084.9 = **23 ms**; Σ idle before gather = 114 thread-ms ≈ 7 ms of window |
| gather (fused copy+drop) | 3085.2→3298.8 = 214 | 16.0% | 1123 memcpy + 188 assemble + 205 munmap + 17 madvise = 1534 | only **7.2 / 16 threads busy** — see below |

The barrier→counts-loop→9.1 GB calloc head of assemble is again ~0 ms (lazy
zero-fill mmap). The main thread's 109 ms munmap at t=3344–3458 is the bench
dropping the returned flat buffer *after* the timer stops — not in-window.

## (a) Did the round-3 fixes land as predicted?

Yes, both:

- **Fused copy+drop gather**: the 114 ms serial munmap tail on the main
  thread is gone from the window. Chunk-buffer frees now appear as 205 ms of
  `kernel: munmap [gather]` CPU spread across the 16 workers, overlapped
  with the copy. Combined gather+free: 163+114 = 277 ms (r3) → 214 ms (r4),
  −63 ms (prediction: ≤ 114, realistic ~64 after accounting for the map-lock
  serialization — spot on).
- **Atomic in-order LPT handout**: straggler spread 104 ms (r3) → 23.4 ms
  (r4) ≈ one ~10 MB tail chunk on an E-core, exactly the predicted bound.
  Residual idle-equivalent is 7 ms of window (0.5%) — no longer a target.

Window: 1462 → 1336 ms (prediction was ~1310; encode-phase CPU also changed
between the traces, walker3 merged in the same mainline).

## (b) Fork + seed ramp (32 ms wall, 373 ms CPU)

Still 16 × 2^22-slot tables = 2 GB of user-side `__bzero` (267 ms of the
289 ms memset CPU is the `alloc_zeroed` under `fork_sized`), plus 84 ms CPU
of vocab seeding (~5 ms/worker). It cannot usefully overlap the serial
prefix: pool spawn + split scan + chunk build measured ~2 ms in r3 (and the
category never accrues visible CPU in r4 either). Memcpy-cloning one seeded
prototype is 2 bytes of traffic per byte vs 1 for memset — strictly worse.
The lever that *is* real: the table is ~2x oversized for a 16-way share.
`fork_sized(640 MB)` sizes linearly (`bytes/256` = 2.5 M → 2^22 slots =
128 MB), but distinct short pretokens follow Heaps' law: calibrating on OWT
(1.3 M @ 1 GB, 5.5 M @ 10 GB ⇒ distinct(n) ≈ 3.45·n^0.62) a 640 MB share
sees ~0.99 M distinct — 2^21 slots (64 MB) holds that at 47% load, and even
a 2.1x-oversubscribed worker stays under the 3/4 growth threshold.
Halving the memset traffic (2 GB → 1 GB) saves ~12–15 ms of ramp.

## (c) P vs E cores, steady state

Per-worker encode CPU spans 846–972 ms over an 1090 ms phase in which every
worker runs continuously (idle condvar CPU < 15 ms/worker until the tail);
the spread is E/P scheduling, as in r3. Average on-CPU utilization is
13.45/16 — the ~2.5-core deficit is threads runnable-but-descheduled
(sampler overhead at 4 kHz × 17 threads, plus macOS housekeeping), not a
code-visible queue. Work-stealing granularity is fine: the in-order handout
bounds the tail at one small chunk (23 ms), so an E-core-aware smaller final
chunk could recover at most ~7 ms of window — not worth the churn (skipped).

Note encode CPU (14.7 s) exceeds the ~11 s ST CPU for the same input: E-core
cycles are slower per unit work, and 16 cold per-worker caches re-derive the
Zipf head 16 times (Σ per-worker distinct ≈ 16 M vs 5.5 M ST) — inherent to
sharded caches, not overhead a scheduler change can recover.

## (d) First-touch faults inside encode

Per-chunk output Vecs are reserved once at `byte_len/4 + 16` (> tokens
needed at OWT's ~4.4 B/token), and no `extend`-growth memcpy shows up in the
encode-phase stacks. The only in-encode memsets are `SpanBatch::new` scratch
(79 ms CPU total, ~5 ms/worker). First touch of the 9.1 GB of fresh output
pages is inherent while results must coexist before the gather; recycling
buffers across calls would trade it for a permanently-held ~9 GB pool —
rejected. Nothing to do here.

## (e) Remaining kernel/lock time — the gather is now a lock convoy

The gather runs at 7.2/16 busy: each worker shows only ~65–120 ms of CPU
(memcpy+munmap+assemble) across the 214 ms phase, and the idle categories
show ~0 — so workers spend ~45% of the phase *blocked*, not spinning. The
copy's first touch of `flat` is ~555k zero-fill faults (vm-map read lock);
the interleaved chunk-buffer frees are munmaps (vm-map **write** lock,
~0.7 ms each × 307). Each munmap stalls every concurrently faulting thread:
a classic reader/writer convoy. Pure page-teardown CPU is only 205 ms — the
phase loses ~1.4 s of thread-time to blocking.

**Fix implemented**: keep the parallel copy, but hand the spent chunk
buffers to a detached `rayon::spawn` that drops them *after*
`assemble_ragged` returns (only when they total ≥ 32 MB). The window then
pays copy+faults only (~1330 ms CPU at r3's ~12+ threads busy ⇒ ~110–160 ms
wall); the ~200 ms of page teardown overlaps the caller's post-processing on
one background worker. Sequential-phase alternatives are strictly worse:
copy-then-parallel-drop still serializes all munmaps on the write lock
(~110+ ms wall, r3's serial free), and pre-faulting `flat` before the copy
just moves the same faults earlier.

## Implemented (this round)

1. **Deferred chunk-buffer drop** (`assemble_ragged`): copy tasks borrow the
   chunks; one detached background task frees them after return. Expected
   saving ~60–100 ms of window (gather 214 → ~120–160 ms).
2. **Heaps-law worker table sizing** (`fork_sized`): 2^21 slots instead of
   2^22 for a 10 GB/16 share. Expected saving ~12–15 ms (ramp 32 → ~20 ms).

Combined expectation: 1336 → ~1240–1270 ms ≈ 7.9–8.1 GB/s.

## Evaluated and rejected (this round)

- Eager pool init / overlap fork+seed with the prefix: prefix is ~2 ms.
- Memcpy- or COW-clone of a seeded prototype table: memcpy is 2x memset's
  traffic; mach_vm_remap is platform-specific for ≤ 20 ms — still deferred.
- E-core-aware smaller tail chunks: bounded upside measured at ~7 ms.
- Output-Vec size hints / double buffering: no growth observed (see (d)).
- Prefault or huge-page the flat buffer: unchanged from r3 (no THP on
  macOS/Apple Silicon; the faults are already the parallel first touch).
