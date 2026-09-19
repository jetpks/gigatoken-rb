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

  it "indexes from the end with a negative index and gives nil out of range, like Array" do
    expect(packed[-1]).to eq([30, 40, 50])
    expect(packed[-3]).to eq([10, 20])
    expect(packed[3]).to be_nil
    expect(packed[-4]).to be_nil
  end

  it "raises TypeError for a non-Integer index, like Array" do
    [nil, "a", 1.5..2.0].each do |index|
      expect { packed[index] }.to raise_error(TypeError)
    end
  end

  it "accepts anything with to_int, like Array" do
    index = Object.new
    index.define_singleton_method(:to_int) { 2 }

    expect(packed[index]).to eq([30, 40, 50])
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
