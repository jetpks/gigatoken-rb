# Zen 5 single-threaded encode profile (Ryzen 7 9800X3D)

Deep `perf` profile of `encode_st` (gpt2, OWT) on the Zen 5 box, splitting the
cold (pass 0, miss-path) and warm (pass >= 1, hit/emit-path) phases. Companion
to `x86_port_plan.md` §6 (which has the A/B throughput history) and
`campaign_report.md` (ARM methodology this replicates with Linux perf).

**Headline: the single biggest finding is not in the instruction stream.** The
pretoken table (and every other large allocation) is running on 4 KiB pages —
`MADV_HUGEPAGE` is issued *after* `alloc_zeroed`'s memset has already faulted
the whole table as small pages, so THP never engages. Fixing the ordering
(measured via a no-code-change glibc A/B) is worth **+15.4% cold / +7.3% warm**
at 1 GB, with the gap growing at full size (the 256 MB full-size table walks
the dTLB even harder). Everything else below is measured on both sides of that
fix.

## 0. Setup

- Box: AMD Ryzen 7 9800X3D (Zen 5, 8C/16T, full AVX-512), 96 MB L3 (X3D),
  L2 1 MB/core, Linux 7.0.0-27-generic, `perf 7.0.6`,
  `kernel.perf_event_paranoid = -1` (nothing blocked).
- Tree: branch `encode-opt-x86-perf` @ `6cc44b2` (worktree
  `.claude/worktrees/agent-a5ba8c6f81c499ac8`, reset to that commit;
  `.cargo/config.toml` `[env] PYO3_PYTHON=...` copied from the main checkout,
  `data/` symlinked).
- Build: `cargo bench --no-run --bench encode_st` → the **bench profile**
  (inherits release: fat LTO, identical codegen, plus full debuginfo). The
  `profiling` profile was deliberately NOT used: it sets `lto = false`, which
  changes the inlining under test. Flat cycle sampling + srcline/annotate
  attribution needs no frame pointers.
  Binary: `target/release/deps/encode_st-57c4b9fdb93c41d6`. Baseline
  runtime-dispatch build (no `-C target-cpu`) — AVX-512 tier, CRC hash and
  probe_pair pin all engage via runtime detection, exactly what wheels ship.
- Token identity verified on **every** profiled run: 228107519 @
  `ENCODE_MB=1000` (all passes), 22834020 @ 100 MB, 2717102153 @ full 11.9 GB.
  No mismatches.
- Baseline throughput reproduced: 682–700 MB/s cold, 1008–1057 MB/s warm @
  1 GB (references: 685 / ~1010); 774 MB/s full-file cold (ref ~763).
- Phase windows at `ENCODE_MB=1000` (timestamped stderr): load+read ends
  ~0.34 s; pass 0 = 0.34–1.79 s; passes 1–5 = 1.79–6.72 s. Hence
  cold = `perf --delay 400` with `ENCODE_PASSES=1`,
  warm = `perf --delay 1900` with `ENCODE_PASSES=6` (window is pure warm).
- All runs sequential, machine idle. Repro commands: `profiling/zen5/repro.sh`
  (every number in this file maps to a line there). Raw perf data + stat logs
  in `profiling/zen5/`.

## 1. Cycle breakdown by function

Fat LTO folds the code into a few giant symbols; the mapping column comes from
srcline profiles + `perf annotate` (loop address ranges), not guesswork:

- `fill_spans_two_phase_crc<R50kScheme, {closure}>` = pretokenize fill: phase A
  (boundary bit algebra + `flatten_bits`) **and** the phase-B per-span emission
  loop (key pack + CRC + L2-stage prefetch + batch-entry store),
  `src/pretokenize/fast/mask.rs`. The phase-B fast loop is the address range
  `0x9ecc0–0x9ed4f`; it is **69.0% of this symbol warm, 70.5% cold** (awk sum
  over annotate).
- `encode_st::main` = the inlined `memoized_encode_flat` **probe/emit chunk
  loop** (`src/bpe/tiktoken.rs` `probe_emit_chunk` + `pretoken_cache.rs`
  `probe_pair`); the emit fast loop `0x47930–0x47a12` is 96–97% of the symbol.
- `batch_masks_avx512` = AVX-512 ASCII classification (`r50k.rs`);
  `extended_masks` = scalar bad-zone/non-ASCII batches (`r50k.rs`).
- `merge_short` = the miss-path short BPE merge incl. inlined
  `PairRankTable::rank` (`src/bpe/mod.rs`).

### Cold (pass 0, 1 GB, 5580 samples, cycles:u)

| % | symbol | maps to |
|---|---|---|
| 33.45 | `fill_spans_two_phase_crc` | fill: 70.5% phase-B span loop (**23.6% of phase**), 29.5% phase A/flatten/drive (9.9%) |
| 28.91 | `encode_st::main` | probe/emit fast loop (96% of it → **27.8% of phase**) |
| 17.83 | `merge_short` | miss-path short merge; ~27% of it = `PairRankTable::rank` sparse-slot probe loads |
| 6.41 | `batch_masks_avx512` | AVX-512 classification |
| 3.77 | `extended_masks` | scalar bad-zone (non-ASCII batches) |
| 2.55 | `probe_emit_slow` | spill/miss handling off the emit loop |
| 2.41 | `bpe_merge_symbols_by_rank<{closure}>` | long-pretoken merge |
| 0.68 | kernel (page faults) | table/output first-touch |
| 0.67 | `ShortPretokenCache::grow` | rehash on growth |
| 0.65 | `encode_pretoken_miss` | miss orchestration + insert |
| 0.57 | libc `__memmove_avx512` | misc copies |
| 0.48 | kernel | faults |
| 0.18 | `fill_spans_keyed_mask` | dispatch shim |
| ~1.4 | rest | < 0.15% each |

Cold srcline top: `mask.rs:1104` 11.2% (phase-B entry store), `pretoken_cache.rs:393/394`
10.2% (probe_pair slot deref + compares), **`bpe/mod.rs:225` 5.66%** (`rank()`
sparse-slot compare — i.e. the rank-table load), `tiktoken.rs:822` 5.0% (emit),
`sse.rs:1946` 4.3% (`_mm_prefetch`), `core/ptr/mod.rs:1939` 4.0%
(`read_unaligned` of the 16 B key), `bpe/mod.rs:209` 1.2% (rank dense path).

Full-file (11.9 GB) shape check matches: fill 41.6 / main 30.5 / merge_short
9.6 / batch_masks 6.5 / extended 4.0 / slow 2.2 / by_rank 2.1 — merge_short
share halves because the cache is warmer over a longer pass; nothing new
appears at scale.

### Warm (passes 1–5, 1 GB, 19314 samples, cycles:u)

| % | symbol | maps to |
|---|---|---|
| 46.73 | `fill_spans_two_phase_crc` | 69.0% phase-B span loop (**32.2% of phase**), 31% phase A/flatten (14.5%) |
| 37.50 | `encode_st::main` | probe/emit fast loop (**36.5% of phase**) |
| 7.45 | `batch_masks_avx512` | AVX-512 classification |
| 4.68 | `extended_masks` | scalar bad-zone |
| 2.32 | `probe_emit_slow` | leftover slow-path |
| 0.70 | libc `__memmove_avx512` | |
| 0.22 | libc `__memcmp_evex` | long-map key compare |
| 0.19 | `avx512_scanner_available` | dispatch check (per fill) |

Warm srcline top: `mask.rs:1104` 14.2%, `pretoken_cache.rs:393/394` 14.35%,
`sse.rs:1946` (`_mm_prefetch`) 6.7%, `tiktoken.rs:822` 6.2%, `ptr/mod.rs:1939`
(key `read_unaligned`) 4.5%, `mask.rs:1093–1096` (keep/pack) ~7.9%,
`mask.rs:775/779` (`flatten_bits` BIT_POS store loop) 4.4%, `sse42.rs:21`
(`crc32q`) 1.2%, `pretoken_cache.rs:222` (`get_or_slot`) 0.9%.

With THP forced (see §3) the shape barely moves (fill 43.2 / main 39.6 /
batch_masks 8.6 / extended 5.3 / slow 2.3): the win is spread across both big
loops' memory accesses, and the residual bottleneck ordering is unchanged.

## 2. Top-down / bottleneck characterization

`perf stat -M PipelineL1` / `PipelineL2` (AMD Zen topdown metric groups), plus
miss-rate metrics. Default pages unless marked THP.

| metric | cold (pass 0) | warm (passes 1–5) | warm+THP |
|---|---|---|---|
| IPC (user) | **3.88** (26.45G inst / 6.81G cyc) | **4.83** (122.0G / 25.25G) | **5.23** (112.2G / 21.45G); cold+THP 4.38 |
| retiring | 42.4–42.8% slots | 55.2–55.9% | — |
| frontend bound | 17.2% (latency 12.3 + bw 4.9) | 5.3% (latency 4.0 + bw 0.8) | — |
| backend bound | 31.1–31.8% (**memory 29.8**, cpu 1.7) | 35.6% (**memory 33.3**, cpu 1.9) | — |
| bad speculation | ~8.6% (mispredicts 8.2 + restarts 0.4) | ~3.9% (mispredicts 3.9) | — |
| branch miss rate | 1.9% (37.3M / 1.95G) | 0.72% (58.9M / 8.19G) | — |
| L1d miss rate | 2.6% | 1.7% | — |
| L1i miss rate | 7.2% | 8.2% (tiny absolute: 3.2M vs cold 14.9M) | — |
| L2 hits from L1d miss (pti) | 6.9/k inst | 5.9/k inst | — |
| L2 misses from L1d miss (pti) | 1.4/k inst (38.1M/pass) | 0.2/k inst (5.5M/pass) | — |
| dTLB-loads (L1-dTLB miss) | 256M/pass | 233M/pass (1.165G/5) | **31K/pass (155K/5)** |
| dTLB-load-misses (walks) | 26.2M/pass | 28.6M/pass (143M/5) | **2.3K/pass** |

Reading:

- **Warm is backend/memory bound (33% of slots), not branch or frontend
  bound.** This is the mirror image of the ARM baseline story (25% bad
  speculation); the campaign's de-branching already paid off, and Zen 5's
  predictor handles the residual `rank`/spill branches fine. Retiring 55% at
  IPC 4.8 (5.2 with THP) — the hit path is closing in on issue limits, and the
  remaining slack is almost entirely load latency in the two hot loops.
- **Cold adds two things: mispredicts (8.2% of ops) and frontend latency
  (12.3% of slots, L1i miss rate 7.2%).** Both come from the miss path:
  `merge_short`'s data-dependent merge scan and the hit/miss interleaving
  (the miss path's code footprint evicts the hot loops' lines; 14.9M icache
  misses in one cold pass vs 3.2M across five warm passes).
- **The dTLB story is the big one**: ~28M page walks per pass, warm or cold.
  L2 dTLB (4096 entries × 4 KiB = 16 MiB reach) cannot cover 64 MB table +
  1 GB input + 0.9 GB output. On Zen, software prefetches that miss the dTLB
  are dropped, so the emit loop's `prefetcht0` D=16 stage silently fails on
  exactly the loads it exists to hide (visible: 13.3% of emit-loop samples sit
  on `prefetcht0` with 4 KiB pages, 9.8% with THP). With THP the walks go to
  ~zero and warm IPC gains +0.4.
- L3: not directly measurable (l3 PMU events absent on this kernel/perf), but
  L2-miss traffic is 38M lines cold vs 5.5M warm per pass: the warm 64 MB
  table probe set is effectively L3-resident on the 96 MB X3D cache (DRAM
  traffic minimal); at full size the 256 MB table cannot be.

## 3. THP: the table runs on 4 KiB pages, and why

Observed (`/proc/<pid>/smaps` mid-run, 1 GB warm passes):

```
7d80b0000000-7d80b4000000  size=65536kB anon=65536kB AnonHugePages=0kB THPeligible=1   <- short table
7d80b7f5c000-7d812f2b6000  size=1953128kB anon=1867612kB AnonHugePages=0kB THPeligible=0 <- input+output
```

- sysfs: `enabled=madvise`, `defrag=[madvise]`, `hugepages-2048kB=inherit` — OK.
- The session sets `PR_SET_THP_DISABLE=1`; the bench's `common::allow_thp()`
  prctl clears it (strace-verified, returns 0) — OK. **Any production launcher
  that does not clear it gets nothing, regardless of the code fix below.**
- `madvise(table, MADV_HUGEPAGE)` is called and returns 0 (strace) — OK.
- **Root cause (reproduced standalone in python):** `Slots::new_zeroed`
  (`src/bpe/pretoken_cache.rs:67-73`) calls `alloc_zeroed(layout)` *then*
  `madvise_hugepage(raw, size)`. For a 2 MiB-aligned layout, Rust's System
  allocator implements `alloc_zeroed` as `aligned_alloc` + **explicit
  `write_bytes(0)`** — the memset faults in every page of the fresh mapping as
  4 KiB pages *before* the madvise runs. A VMA madvised *before* first touch
  gets 2 MiB pages at fault time (verified: same mmap + madvise-first +
  identical touch pattern → `AnonHugePages=65536kB`, whether touched
  sequentially, randomly, or read-then-write); madvised *after* touch it stays
  4 KiB and only khugepaged slowly collapses it (~16 MB collapsed in tens of
  seconds in the repro — irrelevant for a 1.5 s pass, marginal for long-lived
  pools). The input/output Vecs are never madvised at all (THPeligible=0
  under `enabled=madvise`).

### Measured value (no-code-change A/B)

`GLIBC_TUNABLES=glibc.malloc.hugetlb=1` makes glibc issue `MADV_HUGEPAGE` on
its mmaps *at creation time* — i.e. exactly the ordering fix, applied to every
big allocation (table + input + output; smaps_rollup shows 1.95 GB
AnonHugePages). Interleaved A/B, 5 pairs, `ENCODE_MB=1000 ENCODE_PASSES=3`,
token counts identical everywhere:

| | cold (pass 0) | warm (passes 1–2) |
|---|---|---|
| default (4 KiB) | 688–700, mean ~695 MB/s | 1011–1057, mean ~1027 MB/s |
| THP | 797–805, mean ~802 MB/s | 1095–1107, mean ~1101 MB/s |
| delta | **+15.4%** | **+7.3%** |

Counter confirmation: warm dTLB-loads 1.165G → 155K, walks 143M → 11K, IPC
4.83 → 5.23; sys time 0.44 → 0.13 s (fault count /512 on first touch).
At full size (256 MB table = 65536 × 4 KiB pages, vs 128 × 2 MiB) the dTLB
argument is strictly stronger; expect the cold full-file 774 MB/s to gain at
least as much.

## 4. Instruction-level annotate of the three hottest regions

Percentages are of the containing symbol's samples (cycles:u, skid applies:
samples pile on the *consumer* of a slow result and on retire-bound stores).

### 4.1 Phase-B span-emission loop (mask.rs `fill_spans_two_phase_crc`, 0x9ecc0–0x9ed4f) — 32.2% of warm cycles

37 scalar instructions per span, IPC-bound with two latency seams:

```
 2.95  9ecc3: movzwl (%rsi), %r14d          ; boundary u16 load (serial walk)
 1.52  9ecca: subq   %r10, %r8              ; tok_len = end - prev
       9eccd: cmpq   $0xf, %r8
       9ecd6: cmovbq %r8, %rdi              ; m = min(tok_len, 15)
 1.96  9ecda: movl   %edi, %r11d
 0.50  9ecdd: shll   $0x4, %r11d
       9ece1: movq   (%rax,%r11), %r9       ; PACK_MASK_TABLE[m] lo  (dep. on m)
 1.81  9ece5: movq   0x8(%rax,%r11), %r11   ; PACK_MASK_TABLE[m] hi
 3.04  9ecea: andq   (%r15,%r10), %r9       ; raw key lo & mask (16B unaligned load folded)
 0.58  9ecee: andq   0x8(%r15,%r10), %r11   ; raw key hi & mask
 2.02  9ecfa: orq    %r11, %rdi             ; khi |= m<<56
 2.10  9ed07: cmovaeq %r11, %r9             ; keep-mask (long-span route), if-converted
 3.16  9ed0b: cmovaeq %r11, %rdi
 2.07  9ed15: crc32q %r9, %r11              ; CRC chain (2 x 3 cyc)
       9ed1b: crc32q %rdi, %r11
 1.87  9ed2b: shlq   $0x5, %r11
 3.58  9ed2f: prefetcht1 (%rcx,%r11)        ; L2 stage of the table line
29.39  9ed34: movq   %rdi, 0x8(%r13)        ; e.key hi store  <- pileup point
 0.91  9ed38: movq   %r9, (%r13)            ; e.key lo
       9ed3c: movq   %r10, 0x10(%r13)       ; e.ptr
 1.05  9ed40: movq   %rbx, 0x18(%r13)       ; e.meta
 1.69  9ed48: addq   $0x20, %r13
 2.07  9ed4f: jne    0x9ecc0
```

The 29.4% single-instruction spike is the store right after `prefetcht1` —
same `e.ptr = p` retire-pileup signature the M4 analysis saw, amplified here
by the prefetch: with 4 KiB pages the prefetch's dTLB walk stalls the
load/store pipes (the spike + the 3.6% on `prefetcht1` shrink under THP).
Per-span critical path: boundary load → sub → min → table row load → and →
crc32×2 → prefetch address. The span-to-span carried dependency is only
`prev = end` (one u16 load), so the loop overlaps across spans and runs near
issue width — like the M4, restructuring won't help; removing instructions or
the prefetch stall will.

### 4.2 Probe/emit fast loop (`memoized_encode_flat` inlined into main, 0x47930–0x47a12) — 36.5% of warm cycles

~45 instructions/pretoken. The four sample spikes are all one thing — the
random table-line load latency — plus a register-pressure artifact:

```
 9.58  4793b: movq   0x68(%rsp), %r11       ; RELOAD spilled table mask (loop-invariant!)
 0.58  47940: addq   $0x20, %r12            ; next entry
       4794d: movq   %r11, 0x68(%rsp)       ; ...and re-spill it
 0.14  47952: andq   $-0x2, %r11            ; pair index = h & mask & ~1
 0.19  47956: movq   0x218(%r12), %rcx      ; entries[i+16].meta  (D=16 lookahead)
13.28  47965: prefetcht0 (%rax,%rcx)        ; L1 promote, D=16 ahead
 1.55  47969: movq   (%r12), %rcx           ; key lo
 0.10  47972: movdqa (%r12), %xmm0          ; key (u128)
 0.08  47978: movq   0x18(%r12), %r9        ; h = meta
 0.25  4797d: andq   %r9, %r11
 0.03  47984: movdqa (%rax,%r11), %xmm1     ; e0.key  <- THE table load
18.24  4798a: pcmpeqb %xmm0, %xmm1          ; consumer of it (skid target #1)
 1.67  4798e: pmovmskb %xmm1, %edi
 2.04  47995: cmpl   $0xffff, %edi ; sete %r10b   ; m0
 0.66  4799f: pcmpeqb 0x20(%rax,%r11), %xmm0     ; e1.key compare
 0.95  479a6: pmovmskb %xmm0, %edi
14.71  479b0: sete   %bl                    ; m1 (skid target #2)
 0.98  479b3: movq   0x30(%rax,%r11), %rdi  ; e1.val } four value words
 0.04  479b8: movq   0x38(%rax,%r11), %r15  ; e1.ext } loaded up front
 1.35  479bd: movq   0x10(%rax,%r11), %rdx  ; e0.val } (the probe_pair pin,
       479c2: movq   0x18(%rax,%r11), %r11  ; e0.ext }  survives fat LTO)
 0.54  479c7: testq  %r10, %r10
 0.10  479ca: cmovneq %rdx, %rdi            ; val: register-value select
 0.01  479ce: cmovneq %r11, %r15            ; ext
14.11  479d2: movl   %edi, %edx             ; consumer of cmov'd val (skid #3)
 1.04  479d4: shrl   $0x8, %edx             ; ab lane pack
 0.68  479e4: andq   %r14, %r11 ; orq %rdx, %r11
 0.01  479f5: movq   %r11, (%rsi,%r14,4)    ; token stores (2 x 8B)
10.23  479f9: movq   %r15, 0x8(%rsi,%r14,4) ; (skid #4: store retire pileup)
 1.64  479fe: je     0x47a13                ; fast predicate branch 1 (key==0)
 0.03  47a00: orb    %bl, %r10b ; je 0x47a13 ; branch 2 (!found)
 0.11  47a07: andl   $0x80, %edx ; je 0x47930 ; branch 3 (spill)
```

- 4798a + 479b0 + 479d2 ≈ **47% of the loop = the two-line probe load
  latency** (L2 ~14 cyc / L3 ~47 cyc on the 64 MB table; the L2-miss rate says
  the warm mix is mostly L2-hit + a meaningful L3 fraction). The probe_pair
  cmov pin is present and doing its job; the remaining stall is the load
  itself, which only prefetch distance/quality or footprint can move.
- `prefetcht0` at 13.3% (9.8% under THP): on 4 KiB pages many of these
  dropped on dTLB miss (§3); under THP it still costs an AGU/load-pipe slot
  per pretoken.
- **The 9.6% mask reload is pure register pressure**: the loop-invariant table
  mask lives in `0x68(%rsp)` and is reloaded+re-spilled every iteration, on
  the critical path *ahead* of the probe address computation. The `w` cursor
  round-trips through `0x8(%rsp)` the same way. This is an x86-only artifact
  (aarch64 has 31 GPRs) that the ARM campaign never saw.
- The three-branch `fast` predicate (§3.4 of the plan) is a non-issue: 1.6% +
  0.03% + 0.1% of loop samples, warm frontend-bound 5.3%, branch-miss 0.7%.

### 4.3 `merge_short` (cold only; 17.8% of cold cycles) — dominated by rank probes

```
       13a090: movq  (%r14,%rbx,8), %r12    ; PairRankTable sparse slot load
17.93  13a094: movq  %r12, %rcx             ; consumer (skid): slot >> 21 == key?
 0.80  13a097: shrq  $0x15, %rcx
       13a09b: cmpq  %r11, %rcx ; je hit
 1.01  13a0a4: je    miss                   ; slot == u64::MAX
 2.01  13a0a9: andq  %rsi, %rbx ; jmp back  ; linear probe
...
       13a140: movq  (%r14,%rax,8), %rcx    ; second inlined rank() copy
 8.87  13a144: movq  %rcx, %r11             ; same load-latency signature
```

The two sparse-slot load consumers are 26.8% of `merge_short` ≈ **4.8% of the
cold phase**; the srcline view agrees (`bpe/mod.rs:225` = 5.66% of cold
cycles, `:209` dense path another 1.2%). The rest of `merge_short` is the
mandatory serial scan (stack symbol array bookkeeping at `13a0e5` 4.3%,
`13a029` 3.2%, plus 2.8+2.7+2.4% on the scan's jumps — the mispredict tax).
The scan branch structure is what feeds cold's 8.2%-of-ops mispredicts; §6 of
the plan already measured the AVX min-rank scan port **negative** on this box,
so the lever here is the rank *load*, not the scan.

Cache insert / grow are cheap (0.65% + 0.67% of cold); the miss path's cost
is rank lookups >> scan control >> insert.

## 5. Ranked optimization opportunities

1. **Fix THP ordering (madvise before first touch) — measured +15.4% cold /
   +7.3% warm @ 1 GB; expect more at full size.**
   Evidence: §3 (AnonHugePages=0 with THPeligible=1; standalone ordering
   repro; glibc-tunable A/B; dTLB walks 143M→11K per 5 passes).
   Change: in `Slots::new_zeroed`, allocate **uninit** (`alloc`), call
   `madvise_hugepage`, then `write_bytes(0)` (or `mmap` directly); same
   ordering audit for `Committer::try_new` / `gather_flat` (batch.rs) and
   ideally `MADV_HUGEPAGE` on the flat output reservation and (in the bench)
   the input buffer — the measured A/B had all three huge. Also document that
   PR_SET_THP_DISABLE must be cleared by real launchers (python bindings do
   not run the bench's prctl) and khugepaged is *not* a rescue at these run
   lengths. Cheap, zero-risk, biggest single win available.

2. **Kill the emit loop's loop-invariant spills (mask + cursor) — est. 2–4%
   warm.** Evidence: §4.2 — 9.6% of the hottest loop is reloading a
   loop-invariant mask from the stack, on the probe-address critical path;
   `w` also round-trips through memory. The loop has ~17 live values vs 15
   usable GPRs (xmm0/1 already used). Options: precompute `mask & !1` once
   (removes the per-iteration `andq $-2` and one temp), fold the prefetch
   index math to reuse a dying register, or pin `mask` in a register via a
   one-line `asm!` (the campaign's established pin pattern). Verify with
   `perf annotate` that `0x68(%rsp)` traffic disappears from the loop.

3. **Prefetch stage re-tune for Zen 5, after THP (plan §3.3) — est. 0–3%
   warm.** Evidence: `prefetcht0` still eats 9.8% of emit-loop samples under
   THP and the probe compares still hold ~45%; backend-memory stays 33%.
   With THP the table line's TLB entry is nearly always present, so the D=16
   L1-promote may now be too *short* (issue→use ≈ 45 pretoken-loop cycles?
   measure) or redundant with the phase-B `prefetcht1` stage. Sweep D ∈
   {8, 16, 24, 32} and A/B dropping either stage (T1-only / T0-only /
   both) — compile-time consts, cheap to run under the §3 protocol.

4. **Cold miss path: hide the `PairRankTable::rank` sparse-probe latency —
   est. 2–4% cold.** Evidence: §4.3 — ~6–7% of cold cycles are the sparse
   slot load consumers; the AVX scan port already measured −1% (plan §6), so
   attack the load: (a) issue a `prefetcht0` for the *next* pair's slot
   before evaluating the current pair in the merge scan (the address depends
   only on `(sym[i+1], sym[i+2])`, available one step early); (b) widen
   `dense_log2` (currently `clamp(8,10)` → 4 MB max dense table; GPT-2's hot
   pair IDs are heavily < 2^11–2^12 — a 16 MB dense table at log2=11 fits the
   X3D L3 easily; A/B it); (c) 4-byte fingerprint-in-index packing to halve
   probe line traffic. Also helps MT (16 cold caches pay this per worker).

5. **AVX-512 `vpcompressb` for `flatten_bits` (plan §3.5's real successor) —
   est. 1–2% warm.** Evidence: `mask.rs:775/779` = 4.4% of warm cycles (the
   8-octet BIT_POS LUT store loop + per-octet `rel` re-splat). Zen 5 runs
   `vpcompressb` natively at 512-bit: `positions = vpcompressb(iota64, kmask)`
   then widen+`vpaddw` the base — replaces the whole LUT walk for a 64-bit
   mask in ~6 instructions. This is the first x86 box where this is worth
   trying (Zen 2 target has no AVX-512; the plan's §3.5 splat hoist is the
   AVX2 fallback version).

6. **AVX-512 masked key load in phase B — est. 1–3% warm, needs A/B.**
   Evidence: §4.1 — per span, the PACK_MASK_TABLE row load (dependent on
   `m`), two `andq`-with-memory, the `keep` cmovs and the `m<<56` merge are
   ~9 of 37 instructions. With VBMI/AVX-512BW: `k = bzhi(0xffff, m)`;
   `vmovdqu8 (p){k}{z}` gives the masked 16-byte key in one op (the length
   byte can ride in via `vpinsrb`/`or` on the extracted half). Cost: 2×
   `vmovq`/`vpextrq` to feed `crc32q` (GPR-domain) — that crossing is why
   this needs measurement, but it removes the table-row load from the
   critical path entirely. Keep the store layout identical.
7. **Cold frontend (12.3% latency-bound, L1i 7.2%): try PGO / hot-cold
   splitting — est. 1–3% cold.** Evidence: §2 — cold-only frontend latency
   with 14.9M icache misses/pass, caused by hit-loop ↔ miss-path
   interleaving at pretoken granularity. `#[cold]`/`#[inline(never)]` are
   already used on the slow paths; the remaining lever is layout (PGO or
   `-Z function-sections` + linker order file). Low effort to try with the
   existing bench as the profile source.

Explicitly **not** worth pursuing on this box (negative/zero evidence):
- Folding the emit loop's 3-branch `fast` predicate (plan §3.4): warm
  frontend-bound is 5.3%, the branches are <2% of loop samples and predicted
  (0.72% overall miss rate). Zen 5's predictor eats this shape.
- SIMD min-rank scan in `merge_short`: already measured −1% on this box
  (plan §6); the profile confirms the scan is latency-, not compare-bound.
- Walker/phase-A restructuring: phase A + classification ≈ 22% warm but is
  the same irreducible-chain structure proven at its floor on ARM
  (campaign §5); `batch_masks_avx512` at 7.5% is already the k-register tier.

## 6. Answers to the specific questions posed

- **(a) 3-branch `fast` predicate frontend pressure?** No. Warm
  frontend-bound 5.3% (latency 4.0), branch-miss 0.72%, the three `je`s
  ≈ 1.8% of loop samples combined. Skip §3.4.
- **(b) Is D=16 right for Zen 5?** Unproven either way — but the prefetch
  *mechanism* was broken until THP is fixed (Zen drops SW prefetches on dTLB
  miss; 13.3% of loop samples sat on `prefetcht0` at 4 KiB pages, 9.8% after).
  Re-tune only after THP (opportunity #3); with the table effectively
  L3-resident warm on X3D, T0-only at larger D is the shape to test first.
- **(c) Does the scalar bad-zone path dominate more than on ARM?** No.
  `extended_masks` = 4.7% warm / 3.8% cold / 4.0% full-size, plus its share
  feeding `batch_masks_avx512` (7.5% warm). It is real but fourth-order;
  the AVX-512 tier keeps classification cheap.
- **(d) Where does the cold miss path spend?** `merge_short` 17.8% of cold
  (1 GB), of which ~27% is `PairRankTable::rank` sparse-slot load latency
  (≈5–6% of cold cycles, srcline-confirmed); the serial merge scan +
  its mispredicts are most of the rest; long merges
  (`bpe_merge_symbols_by_rank`) 2.4%; `probe_emit_slow` 2.6%; cache insert +
  `grow` ≈ 1.3% combined — inserts are cheap, rank loads are not.
- **(e) AVX-512-only opportunities?** `vpcompressb` flatten_bits (#5),
  masked key load in phase B (#6). Gathers (`vpgatherqq`) for the probe are
  not attractive: the probe is latency- not throughput-bound and Zen 5
  gathers don't shorten latency. The AVX-512 classification tier is already
  engaged and cheap.

## 7. Reproduction

Everything scripted in `profiling/zen5/repro.sh`; raw artifacts in
`profiling/zen5/` (`cold.data`, `warm.data`, `warm_thp.data`,
`cold_full.data`, `*_pipel1/2.txt`, `*_rates.txt`, `*_l2.txt`). Key one-liners:

```sh
cargo bench --no-run --bench encode_st       # bench profile = release codegen + debuginfo
BIN=target/release/deps/encode_st-57c4b9fdb93c41d6

# phases (1 GB): cold = pass 0 (0.34-1.79s), warm = passes 1-5 (1.79-6.72s)
perf record -e cycles:u -F 4000 --delay 400  -- env ENCODE_MB=1000 ENCODE_PASSES=1 $BIN   # cold
perf record -e cycles:u -F 4000 --delay 1900 -- env ENCODE_MB=1000 ENCODE_PASSES=6 $BIN   # warm
perf stat -D 400  -M PipelineL1 -- env ENCODE_MB=1000 ENCODE_PASSES=1 $BIN                # topdown cold
perf stat -D 1900 -M PipelineL2 -- env ENCODE_MB=1000 ENCODE_PASSES=6 $BIN                # topdown warm
# THP A/B (interleave 5x, sequential):
GLIBC_TUNABLES=glibc.malloc.hugetlb=1 ENCODE_MB=1000 ENCODE_PASSES=3 $BIN
```

Token counts asserted every run: 228107519 (1 GB), 2717102153 (full).
Protocol per campaign rules: sequential only, interleaved A/B within one
session, never compare across sessions (§6 rebase note's thermal warning
held here too).
