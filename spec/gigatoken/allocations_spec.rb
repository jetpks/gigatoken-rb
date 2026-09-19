# frozen_string_literal: true

require_relative "../spec_helper"

# Allocation budgets: what each public operation costs in Ruby objects
# beyond its own result. Counts are GC.stat, exact. Everything a measured
# block touches is bound to a local first, so the count is the library's,
# not the example's. Rust-side allocations don't show here.
RSpec.describe "allocations" do
  # Objects allocated per call of the block, averaged over +times+ calls.
  def allocations(times = 20, &block)
    times.times(&block)
    GC.start
    before = GC.stat(:total_allocated_objects)
    times.times(&block)
    (GC.stat(:total_allocated_objects) - before) / times.to_f
  end

  let(:tokenizer) { Gigatoken::Tokenizer.from_encoding("cl100k_base") }
  let(:docs) { Array.new(50) { |i| "document #{i}: the quick brown fox jumps over the lazy dog. " * 8 } }

  it "encodes and decodes in one object each, the result" do
    tok = tokenizer
    text = docs.first
    ids = tok.encode(text)
    expect(allocations { tok.encode(text) }).to be < 2
    expect(allocations { tok.decode(ids) }).to be < 2
  end

  it "encodes a batch in one Array per document, plus the result and the input snapshot" do
    tok = tokenizer
    batch = docs
    expect(allocations { tok.encode_batch(batch) }).to be < batch.size + 3
  end

  it "encodes a packed batch in a fixed handful of objects however many documents" do
    tok = tokenizer
    batch = docs
    expect(allocations { tok.encode_batch(batch, packed: true) }).to be < 10
  end

  it "unpacks a packed document in one object, the Array, and all of them in one per document" do
    packed = tokenizer.encode_batch(docs, packed: true)
    expect(allocations { packed[3] }).to be < 2
    expect(allocations { packed.to_a }).to be < packed.size + 2
  end

  it "encodes files packed in a fixed handful of objects" do
    tok = tokenizer
    path = File.join(__dir__, "..", "fixtures", "docs.txt")
    expect(allocations { tok.encode_files(path, separator: "<|endoftext|>", packed: true) }).to be < 20
  end

  # The Tokenizer and its native tokenizer, plus on Ruby 3.3 a Hash per
  # keyword-argument hop (3.4+ passes them without one). `load` by name adds
  # only the path copy `File.exist?` makes while deciding the source's shape;
  # the Hub client it no longer builds was five objects on its own.
  it "loads a packaged encoding by name without building a Hub client" do
    direct = allocations(3) { Gigatoken::Tokenizer.from_encoding("cl100k_base") }
    by_name = allocations(3) { Gigatoken::Tokenizer.load("cl100k_base") }
    expect(direct).to be < 5
    expect(by_name - direct).to be < 2
  end
end
