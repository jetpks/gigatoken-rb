#!/usr/bin/env bash
# Zen 5 (Ryzen 7 9800X3D) single-threaded encode profiling — reproduction script.
# Run from the repo root (branch encode-opt-x86-perf @ 6cc44b2). Sequential only.
# Binary: baseline runtime-dispatch build (no -C target-cpu), bench profile
# (= release codegen w/ fat LTO + debuginfo):
#   cargo bench --no-run --bench encode_st
# Adjust BIN to the hash cargo prints.
set -euo pipefail
BIN=./target/release/deps/encode_st-57c4b9fdb93c41d6
OUT=profiling/zen5

# Phase map at ENCODE_MB=1000 (measured): tokenizer load+read ends ~0.34s,
# pass 0 (cold) 0.34-1.79s, passes 1..5 (warm) 1.79-6.72s.
# => cold window: --delay 400 with ENCODE_PASSES=1
# => warm window: --delay 1900 with ENCODE_PASSES=6

# 0) Sanity (token counts must be 228107519 @ ENCODE_MB=1000; 2717102153 full file)
ENCODE_MB=1000 ENCODE_PASSES=3 $BIN

# 1) Cycle profiles
perf record -o $OUT/cold.data -e cycles:u -F 4000 --delay 400  -- env ENCODE_MB=1000 ENCODE_PASSES=1 $BIN
perf record -o $OUT/warm.data -e cycles:u -F 4000 --delay 1900 -- env ENCODE_MB=1000 ENCODE_PASSES=6 $BIN
perf report -i $OUT/cold.data --stdio --percent-limit 0.5
perf report -i $OUT/warm.data --stdio -s srcline --percent-limit 0.8
perf annotate -i $OUT/warm.data --stdio -s encode_st::main   # emit loop @ 0x47930-0x47a12
# fill/phase-B loop lives in the big fill_spans_two_phase_crc symbol @ 0x9ecc0-0x9ed4f
# merge_short rank probes @ 0x13a090/0x13a140 (cold.data)

# 2) Topdown + rates (AMD PipelineL1/L2 metric groups)
perf stat -D 400  -M PipelineL1 -- env ENCODE_MB=1000 ENCODE_PASSES=1 $BIN
perf stat -D 1900 -M PipelineL1 -- env ENCODE_MB=1000 ENCODE_PASSES=6 $BIN
perf stat -D 400  -M PipelineL2 -- env ENCODE_MB=1000 ENCODE_PASSES=1 $BIN
perf stat -D 1900 -M PipelineL2 -- env ENCODE_MB=1000 ENCODE_PASSES=6 $BIN
perf stat -D 1900 -M l1d_miss_rate,dtlb_miss_rate,itlb_miss_rate,l1i_miss_rate -- env ENCODE_MB=1000 ENCODE_PASSES=6 $BIN
perf stat -D 1900 -M l2_cache_hits_from_l1_dc_miss_pti,l2_cache_misses_from_l1_dc_miss_pti -- env ENCODE_MB=1000 ENCODE_PASSES=6 $BIN
perf stat -D 400  -e cycles:u,instructions:u,branches:u,branch-misses:u,L1-dcache-loads:u,L1-dcache-load-misses:u,dTLB-loads:u,dTLB-load-misses:u -- env ENCODE_MB=1000 ENCODE_PASSES=1 $BIN
perf stat -D 1900 -e cycles:u,instructions:u,branches:u,branch-misses:u,L1-dcache-loads:u,L1-dcache-load-misses:u,dTLB-loads:u,dTLB-load-misses:u -- env ENCODE_MB=1000 ENCODE_PASSES=6 $BIN

# 3) THP check (bench must be running; table is the 64 MiB anon region)
#    ENCODE_MB=1000 ENCODE_PASSES=6 $BIN &  then:
#    grep AnonHugePages /proc/<pid>/smaps_rollup
#    awk per-VMA THPeligible/AnonHugePages over /proc/<pid>/smaps

# 4) THP A/B (glibc madvises its mmaps BEFORE they are touched):
for i in 1 2 3 4 5; do
  ENCODE_MB=1000 ENCODE_PASSES=3 $BIN 2>&1 | grep '^pass'
  GLIBC_TUNABLES=glibc.malloc.hugetlb=1 ENCODE_MB=1000 ENCODE_PASSES=3 $BIN 2>&1 | grep '^pass'
done

# 5) Full-size shape check (11.9 GB, read ends ~3.2s)
perf record -o $OUT/cold_full.data -e cycles:u -F 1000 --delay 3400 -- env ENCODE_PASSES=1 $BIN
