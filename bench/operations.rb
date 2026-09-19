#!/usr/bin/env ruby
# frozen_string_literal: true

# Operation benchmark: every public Tokenizer operation on cl100k_base, with
# tiktoken_ruby's equivalent beside it where one exists, reported as one
# table: iterations per second (benchmark-ips), Ruby objects allocated per
# call (GC.stat, exact) and Ruby-heap bytes malloc'd per call. Texts are
# built once outside the loops, so the counts are the library's, not the
# caller's. Rust-side allocations don't show in Ruby's counters (this gem
# routes them through mimalloc); they show in the time column.
#
#   ruby -Ilib bench/operations.rb
#   BENCH_QUICK=1 ruby -Ilib bench/operations.rb

require "benchmark/ips"
require "tiktoken_ruby"
require "tmpdir"
require "gigatoken"

Warning[:experimental] = false # IO::Buffer

QUICK = ENV["BENCH_QUICK"]
DOCS = QUICK ? 200 : 1_000

# Deterministic English-ish prose with a little punctuation and non-ASCII:
# enough vocabulary that the pretoken cache doesn't turn every call into a
# lookup, and stable across runs so numbers compare.
WORDS = %w[the quick brown fox jumps over lazy dog tokenizer allocation
  benchmark ruby rust encode decode document café naïve 日本語 batch buffer
  object string integer parallel worker cache budget release version].freeze

def prose(bytes, seed)
  rng = Random.new(seed)
  out = +""
  out << WORDS[rng.rand(WORDS.size)] << (rng.rand(12).zero? ? ". " : " ") while out.bytesize < bytes
  out.freeze
end

short = "The quick brown fox jumps over the lazy dog. "
medium = prose(2_000, 1)
large = prose(200_000, 2)
batch = Array.new(DOCS) { |i| prose(1_000, 100 + i) }.freeze

tok = Gigatoken::Tokenizer.from_encoding("cl100k_base")
enc = Tiktoken.get_encoding("cl100k_base")
[short, medium, large].each { |t| tok.encode(t) && enc.encode(t) }
ids = tok.encode(medium)
packed = tok.encode_batch(batch, packed: true)
corpus = File.join(Dir.mktmpdir("gigatoken-bench"), "docs.txt")
File.write(corpus, batch.join("<|endoftext|>"))

# operation => [gigatoken call, tiktoken_ruby call or nil]
OPERATIONS = {
  "encode, 45 B" => [-> { tok.encode(short) }, -> { enc.encode(short) }],
  "encode, 2 KB" => [-> { tok.encode(medium) }, -> { enc.encode(medium) }],
  "encode, 200 KB" => [-> { tok.encode(large) }, -> { enc.encode(large) }],
  "decode, 2 KB of ids" => [-> { tok.decode(ids) }, -> { enc.decode(ids) }],
  "encode_batch, #{DOCS} x 1 KB" => [-> { tok.encode_batch(batch) }, -> { batch.map { |d| enc.encode(d) } }],
  "encode_batch packed, #{DOCS} x 1 KB" => [-> { tok.encode_batch(batch, packed: true) }, nil],
  "PackedResult#[], one 1 KB doc" => [-> { packed[7] }, nil],
  "PackedResult#to_a, #{DOCS} docs" => [-> { packed.to_a }, nil],
  "encode_files, #{DOCS} x 1 KB" => [-> { tok.encode_files(corpus, separator: "<|endoftext|>") }, nil],
  "encode_files packed, #{DOCS} x 1 KB" => [-> { tok.encode_files(corpus, separator: "<|endoftext|>", packed: true) }, nil],
  "Tokenizer.from_encoding" => [-> { Gigatoken::Tokenizer.from_encoding("cl100k_base") }, nil],
  "Tokenizer.load(\"cl100k_base\")" => [-> { Gigatoken::Tokenizer.load("cl100k_base") }, nil]
}.freeze

module Measure
  # Objects allocated per call, averaged over enough calls to fit in about
  # half a second (one-off allocations outside the call then don't count).
  def self.objects_per_call(call)
    call.call
    started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    call.call
    once = Process.clock_gettime(Process::CLOCK_MONOTONIC) - started
    times = (0.5 / once).clamp(3, 10_000).to_i
    GC.start
    before = GC.stat(:total_allocated_objects)
    times.times { call.call }
    (GC.stat(:total_allocated_objects) - before) / times.to_f
  end

  # Ruby-heap bytes one call malloc's, with GC off so nothing is reclaimed
  # mid-call.
  def self.malloc_per_call(call)
    call.call
    GC.start
    GC.disable
    before = GC.stat(:malloc_increase_bytes)
    call.call
    bytes = GC.stat(:malloc_increase_bytes) - before
    GC.enable
    bytes
  end
end

def commas(number)
  number.round.to_s.reverse.scan(/\d{1,3}/).join(",").reverse
end

def kib(bytes)
  (bytes < 1024) ? bytes.to_s : "#{commas(bytes / 1024.0)} KiB"
end

report = Benchmark.ips do |x|
  x.quiet = true
  x.config(time: QUICK ? 0.5 : 2, warmup: QUICK ? 0.2 : 1)
  OPERATIONS.each do |name, (fast, reference)|
    x.report("#{name} gigatoken", &fast)
    x.report("#{name} tiktoken_ruby", &reference) if reference
  end
end
ips = report.entries.to_h { |entry| [entry.label, entry.ips] }

puts "| Operation | gigatoken i/s | tiktoken_ruby i/s | gigatoken objects/call | tiktoken_ruby objects/call | gigatoken malloc/call |"
puts "|---|---|---|---|---|---|"
OPERATIONS.each do |name, (fast, reference)|
  row = [name, commas(ips.fetch("#{name} gigatoken")),
    reference ? commas(ips.fetch("#{name} tiktoken_ruby")) : "—",
    commas(Measure.objects_per_call(fast)),
    reference ? commas(Measure.objects_per_call(reference)) : "—",
    kib(Measure.malloc_per_call(fast))]
  puts "| #{row.join(" | ")} |"
end
puts
puts "#{RUBY_DESCRIPTION}; gigatoken #{Gigatoken::VERSION}; tiktoken_ruby #{Gem.loaded_specs["tiktoken_ruby"].version}; #{DOCS} docs per batch"
