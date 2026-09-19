#!/usr/bin/env ruby
# frozen_string_literal: true

# Reproducible evidence (I05/I06) for the claim behind `encode_contended`'s
# `#[cold] #[inline(never)]` pair in ext/gigatoken/src/tokenizer.rs: that
# outlining the contended half of `BPETokenizer#encode` is what keeps the
# hot, uncontended path (`try_write` succeeds, three lines, never releases
# the GVL) free of an inlining-driven regression under this workspace's
# `lto = "fat"`. Under the GVL a fast-path `#encode` never yields, so two
# `#encode` calls are never inside the extension at once — the contended
# path is only reachable from a batch-holding reader, which makes it
# unreachable in a *single-threaded* benchmark. Measuring single-threaded is
# therefore the right test for an inlining artifact, not a lock-contention
# one, and this script only ever calls single `#encode`, never
# `encode_batch`/`encode_files`.
#
# `#[cold]`/`#[inline(never)]` are compile-time attributes with no
# Ruby-visible switch (and this repo's lane rules forbid inventing one in
# lib/ or the extension), so this script cannot flip them at runtime: its
# built-in "A" and "B" arms both run whatever build is currently installed
# in lib/gigatoken/gigatoken_rb.bundle. That is enough to validate the
# *methodology* (interleaving, a self-derived noise floor, distinguishing a
# real delta from noise) against itself; it is not evidence that the
# attributes matter. For that, rebuild with them stripped and rerun — see
# "Counterfactual procedure" below, and docs/explanation/benchmarks.md for the
# numbers from the one time this was done.
#
# The reported statistic is the **median** of per-round samples, not the
# mean. Per-round times on a shared machine are heavily right-skewed — a
# scheduler hiccup costs one round several times its usual cost — and the
# mean lets that single round move the whole result by double digits; the
# median doesn't move until a majority of samples shift. See
# docs/explanation/benchmarks.md's "How much of this can you trust" for the
# comparison across candidate estimators that this choice is based on.
#
# Each round measures arm A *twice* (A1, A2) and arm B once, rotating which
# of the three runs first/second/third across rounds. A1 is what's reported
# and compared against B; A2 exists solely to feed the A/A noise floor. The
# floor itself is a bootstrap: resample two `n`-sized groups (with
# replacement) from the pooled A1+A2 samples many times and take a high
# percentile of their median deltas, rather than a single A1-vs-A2 point
# comparison — a point comparison can land on an exact tie by chance (floor
# 0%) and then call any nonzero noise a real effect. Both the floor and the
# delta it gates are built from `n`-sized samples of the same underlying
# distribution, so comparing them means something. This replaces the earlier
# design, which split A1's own samples in half by round parity — half the
# delta's sample count, and confounded with position since parity also set
# which arm ran first.
#
#   ruby -Ilib bench/encode_ab.rb
#   GIGATOKEN_AB_ROUNDS=1 ruby -Ilib bench/encode_ab.rb   # cheap smoke run
#
# Counterfactual procedure (manual, not automated by this script):
#   1. In ext/gigatoken/src/tokenizer.rs, delete the `#[cold]` and
#      `#[inline(never)]` lines directly above `fn encode_contended`.
#   2. `bundle exec rake compile` (rebuilds lib/gigatoken/gigatoken_rb.bundle).
#   3. `ruby -Ilib bench/encode_ab.rb` — record the "A" column; that's the
#      un-outlined build's single-encode cost.
#   4. Revert step 1's edit exactly, then `bundle exec rake compile` again
#      to restore the shipped build before trusting any other measurement.

require "gigatoken"

ROUNDS = Integer(ENV.fetch("GIGATOKEN_AB_ROUNDS", 80))

SIZES = {
  "short" => [8000, "The quick brown fox jumps over the lazy dog. "],
  "medium" => [40, (["lorem ipsum dolor sit amet, consectetur adipiscing elit. "] * 40).join.freeze],
  "large" => [16, (["lorem ipsum dolor sit amet, consectetur adipiscing elit. "] * 4000).join.freeze]
}.freeze

tok = Gigatoken::Tokenizer.from_encoding("cl100k_base")
# Warm every size's pretoken cache before any timed round. Warming only the
# large text (as an earlier revision did) leaves each other size's very
# first sample to eat a one-time cache-population cost that never recurs —
# and because that first sample always lands on whichever arm runs first in
# round 0, it's a systematic bias against one specific arm, not noise.
SIZES.each_value { |(_, text)| tok.encode(text) }

# Mean seconds per #encode call over `iterations` calls against `text`.
def time_encode(tok, text, iterations)
  start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  iterations.times { tok.encode(text) }
  (Process.clock_gettime(Process::CLOCK_MONOTONIC) - start) / iterations
end

def median(samples)
  sorted = samples.sort
  mid = sorted.size / 2
  sorted.size.odd? ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2.0
end

# Median absolute deviation from `center` — a robust spread to report
# alongside the median, so dispersion is visible rather than inferred.
def mad(samples, center)
  median(samples.map { |s| (s - center).abs })
end

BOOTSTRAP_TRIALS = 2000
BOOTSTRAP_PERCENTILE = 0.98

# A conservative estimate of the spread an A/A comparison exhibits at this
# sample size: `pool` is A1 and A2 combined (two independent samples of the
# same build), and each trial draws two fresh `n`-sized samples from it with
# replacement and takes their median delta. The 98th percentile of those
# trial deltas is the floor. A single A1-vs-A2 point comparison can land on
# an exact tie (floor 0%) purely by chance and then call any nonzero noise a
# real effect; resampling the whole pool many times is what makes the floor
# represent the run's actual dispersion instead of one lucky draw.
def bootstrap_floor(pool, n, trials: BOOTSTRAP_TRIALS, percentile: BOOTSTRAP_PERCENTILE)
  deltas = Array.new(trials) do
    x_med = median(Array.new(n) { pool.sample })
    y_med = median(Array.new(n) { pool.sample })
    (y_med - x_med).abs / x_med
  end
  deltas.sort[(deltas.size * percentile).to_i]
end

def format_us(seconds)
  format("%.2fus", seconds * 1_000_000)
end

def format_pct(ratio)
  format("%+.2f%%", ratio * 100)
end

# All six orderings of the three per-round measurements, so across any six
# consecutive rounds each of A1/A2/B has run first, second and third exactly
# once — no arm is systematically advantaged by running first (or last).
ARM_ORDERS = %i[a1 a2 b].permutation.to_a.freeze

samples = SIZES.each_with_object({}) { |(size, _), h| h[size] = {a1: [], a2: [], b: []} }

ROUNDS.times do |round|
  order = ARM_ORDERS[round % ARM_ORDERS.size]
  SIZES.each do |size, (iterations, text)|
    order.each do |arm|
      samples[size][arm] << time_encode(tok, text, iterations)
    end
  end
end

puts "GIGATOKEN_AB_ROUNDS=#{ROUNDS}"
puts

SIZES.each_key do |size|
  a1 = samples[size][:a1]
  a2 = samples[size][:a2]
  b = samples[size][:b]

  a_med = median(a1)
  b_med = median(b)

  puts "== #{size} =="
  puts "  A (current build):  #{format_us(a_med)}/encode (MAD #{format_us(mad(a1, a_med))})"
  puts "  B (current build):  #{format_us(b_med)}/encode (MAD #{format_us(mad(b, b_med))})"

  floor = bootstrap_floor(a1 + a2, a1.size)
  delta = (b_med - a_med) / a_med
  verdict = if delta.abs < floor
    "indistinguishable from noise"
  else
    delta.negative? ? "faster" : "slower"
  end
  puts "  A/A noise floor:     #{format_pct(floor)}"
  puts "  A/B delta:           #{format_pct(delta)} (#{verdict})"
  puts
end

puts "Both A and B above ran the build currently installed in lib/gigatoken " \
     "(the shipped, #[cold]-outlined encode_contended) — see this file's " \
     "header for why a true attribute-removed B arm needs a manual rebuild. " \
     "\"A\" is measured twice per round (A1 is the column above; its twin " \
     "A2 is held back to build the A/A noise floor) so the floor and the " \
     "delta it gates are computed from the same number of samples."
