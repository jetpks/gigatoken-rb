# frozen_string_literal: true

require_relative "../spec_helper"

RSpec.describe Gigatoken::PackedResult do
  let(:buffer) { IO::Buffer.for([10, 20, 30, 40, 50].pack("L*").b.freeze) }
  let(:lens) { [2, 0, 3] }
  let(:packed) { described_class.new(buffer, lens) }

  it "reports the document count and total token count" do
    expect(packed.size).to eq(3)
    expect(packed.token_count).to eq(5)
  end

  it "materializes a document's token ids by index" do
    expect(packed[0]).to eq([10, 20])
    expect(packed[1]).to eq([])
    expect(packed[2]).to eq([30, 40, 50])
  end

  it "enumerates documents in order, as an Enumerable" do
    expect(packed.each.to_a).to eq([[10, 20], [], [30, 40, 50]])
    expect(packed.map(&:size)).to eq([2, 0, 3])
  end

  it "returns an Enumerator when each is called without a block" do
    expect(packed.each).to be_an(Enumerator)
  end

  it "unpacks to the ragged shape via to_a" do
    expect(packed.to_a).to eq([[10, 20], [], [30, 40, 50]])
  end

  it "exposes the underlying buffer and lens" do
    expect(packed.buffer).to eq(buffer)
    expect(packed.lens).to eq(lens)
  end
end
