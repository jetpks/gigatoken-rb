# MT encode profile analysis — samply_mt_round5 (10 GB cold, GPT-2, 16 threads, M4 Max)

Trace: `samply_mt_round5.json.gz` of the round-4 mainline (encode-opt-main
@1ffff3c: deferred chunk-buffer drop, Heaps-law worker table sizing). Run
measured 1.226 s encode = 8159 MB/s. Analysis: `analyze_mt.py` (round-3
tool, unmodified), outputs in this directory. All CPU accounting uses
`threadCPUDelta` (kernel user+system µs).

## The 1226 ms window (t = 1831.4 → 3057.0), bucket by bucket

| phase | wall ms | % | CPU ms | notes |
|---|---|---|---|---|
| worker fork + seed | ~19 | 1.5% | 143 memset + 54 seed | 16 × 64 MB tables (2^21 slots after Heaps sizing) — memset CPU exactly halved from r4's 289 |
| encode steady state | ~1025 | 83.6% | 14,914 + 199 scratch memset | ~14.4/16 threads busy per 25 ms bin; per-worker encode CPU 908–948 ms (tight E/P spread) |
| straggler tail | 2875.3→2887.0 = 11.7 | 1.0% | — | Σ idle before gather ≈ 77 thread-ms ≈ 4.8 ms of window — at the one-tail-chunk LPT bound |
| gather (copy only) | 2887.2→3056.8 = 170 | 13.8% | 2079 memcpy | 12.2/16 threads busy; ZERO munmap in-window — see below |

Post-window (after the timer): the deferred chunk-buffer drop runs 79.9 ms
of munmap on worker T09 at t=3057–3138 (overlapped with the caller, as
designed), and the main thread's 110.8 ms munmap + 14.8 ms madvise at
t=3063–3200 is the bench dropping the returned flat buffer.

## (a) Did round-4's fixes deliver?

Yes, both, on prediction:

- **Deferred drop**: gather 214 ms (r4, fused copy+drop convoy at 7.2/16
  busy) → 170 ms (r5, copy-only at 12.2/16). The munmap write-lock convoy
  is gone from the window; page teardown moved to one background worker
  after return. Prediction was 120–160 ms — actual 170, close but the copy
  did not speed up as much as the busy-thread count implies (next point).
- **Heaps table sizing**: ramp 32 → ~19 ms wall; fork memset CPU 289 → 143
  (predicted ~12–15 ms saving, got ~13).

Window: 1336 → 1226 ms (prediction 1240–1270; slightly better).

## (b) The gather's CPU nearly DOUBLED while its wall shrank

r4: 1123 ms memcpy CPU for the 9.1 GB copy. r5: 2079 ms for the same copy.
The r4 convoy made threads *sleep* (blocked on the vm-map write lock — no
CPU accrued); with the munmaps gone, all 16 threads fault concurrently and
the cost converts into in-kernel fault-path contention (free-page-queue and
pmap lock spinning is CPU-visible). 2079 ms / 9.1 GB = 228 ms CPU per GB of
first-touch+copy at 12–16-way concurrency vs ~60–100 ms/GB uncontended —
i.e. **more than half of the gather's CPU is concurrency-induced kernel
contention**, on top of ~555k unavoidable 16 KB zero-fill faults (no THP
on Apple Silicon).

## (c) Fork ramp, tail, P/E spread — all at their floors

- Ramp (~19 ms): each worker forks its own table lazily inside its first
  `with_worker` call and starts encoding immediately after — the ramp IS
  task-0 latency, already fully overlapped per-worker; the serial prefix
  before the pool starts is ~0.2 ms. Only a COW-prototype table clone
  (mach_vm_remap, platform-specific) could shrink it further — still not
  worth it for ≤ 19 ms.
- Tail (11.7 ms spread, 4.8 ms window-equivalent): exactly one ~10 MB tail
  chunk on an E-core, the bound in-order LPT handout guarantees.
- Per-worker encode CPU 908–948 ms: the E/P spread is even; no scheduling
  pathology.

## (d) Verdict: orchestration is NOT quite done — one lever left

Non-steady overhead = 19 (ramp) + 5 (tail) + 170 (gather) ≈ **194 ms
(15.8% of window)** — well above the ~80 ms "done" threshold, and 88% of
it is the gather. Ramp and tail are at their floors; per-worker encode
speed aside, the gather is the only remaining target.

## (e) Implemented this round: commit-the-prefix-during-encode

The r3 blocker ("the destination cannot exist until the last chunk's size
is known") dissolves once the destination is *reserved at an upper bound*:
token count never exceeds input bytes (every emitted token consumes ≥ 1
input byte), so `try_reserve_exact(total_bytes)` bounds the buffer — VA
only, untouched pages cost nothing physical. In-order handout (r3) makes
chunk completion near-sequential, so a finishing worker can commit the
ready prefix with known offsets while the tail still encodes:

1. **Prefix committer** (`Committer` in `src/batch.rs`): workers encode
   chunks exactly as before (atomic in-order pull loop); after finishing a
   chunk, a worker `try_lock`s the commit cursor and copies any completed
   prefix chunks into the reserved flat buffer (bounded drain, ≤ 8 chunks
   per event — no run-long committer thread, contended finishers skip and
   go back to encoding). After the scope joins, the uncommitted suffix is
   copied in parallel (the old gather, now over a small residue), the
   buffer is `set_len` + `shrink_to_fit` (in-place tail trim: probed on
   macOS libmalloc — pointer-stable for 1–8 GB blocks; glibc shrinks
   mmap'd chunks in place via mremap).
   - Fallbacks, both to the classic collect-then-gather path: upper-bound
     reservation failure (e.g. Linux heuristic overcommit refusing 4×VA for
     huge inputs), and cursor overflow (only possible when NFC
     normalization expands bytes — composition-exclusion pathologies).
   - Chunk buffers are still freed on the deferred background task, not
     during encode — an early free would reintroduce r4's munmap-vs-fault
     convoy, this time against encode-phase faults.
2. **Single-chunk no-copy**: a lone chunk's id buffer IS the flat result;
   small (serial-path) inputs skip the gather memcpy entirely.

Why it should pay: the copy's 2.08 s CPU moves inside the 1025 ms encode
phase, and diluting 16 concurrent faulters to ~1 at a time should claw
back a large part of the contention half of (b). Expected: gather 170 →
~10–30 ms residual drain, encode phase extends by (copy CPU)/16 ≈ 65–130 ms
depending on how much contention dilution recovers ⇒ window ~1120–1170 ms
≈ 8.6–8.9 GB/s. Memory profile unchanged (flat prefix is touched during
encode instead of after; peak = chunks + flat either way).

## Evaluated and rejected (this round)

- **Overlap fork+seed with first chunks**: already the case — seeding is
  each worker's own first-task latency (see (c)).
- **mlock/madvise prefault of flat**: mlock zero-fills in one kernel entry
  but holds the vm-map lock for the whole wire — serializes at memset
  bandwidth (~300+ ms for 9.1 GB), worse than the parallel faulting copy.
  No THP / superpages on macOS ARM64; MADV_WILLNEED is a no-op for
  anonymous zero-fill.
- **E-core-aware tail chunks**: bounded upside 4.8 ms.
- **Early per-chunk buffer free (during encode)**: reintroduces the
  write-lock convoy against encode-phase faults; deferred drop already
  moved teardown off-window. Revisit only if peak RSS becomes the metric.
