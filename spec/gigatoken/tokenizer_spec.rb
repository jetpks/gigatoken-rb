# frozen_string_literal: true

require_relative "../spec_helper"

RSpec.describe Gigatoken::Tokenizer do
  fixture_path = File.expand_path("../../tests/fixtures/gpt2_tokenizer.json", __dir__)

  let(:tokenizer) { described_class.from_file(fixture_path) }

  it "encodes known GPT-2 vectors" do
    expect(tokenizer.encode("Hello, world!")).to eq([15496, 11, 995, 0])
    expect(tokenizer.encode("")).to eq([])
  end

  it "round-trips UTF-8 samples through encode and decode" do
    ["plain ascii", "café", "日本語のテキスト", "emoji 😀🎉"].each do |text|
      ids = tokenizer.encode(text)
      expect(tokenizer.decode(ids).force_encoding("UTF-8")).to eq(text)
    end
  end

  it "encodes a batch identically to per-string encode" do
    texts = ["Hello, world!", "", "café", "日本語のテキスト", "a longer sentence for batching."]
    expect(tokenizer.encode_batch(texts)).to eq(texts.map { |t| tokenizer.encode(t) })
  end

  it "reports the vocab size and decodes to a BINARY-encoded String" do
    expect(tokenizer.vocab_size).to eq(50257)
    expect(tokenizer.decode([15496]).encoding).to eq(Encoding::ASCII_8BIT)
  end

  it "raises Gigatoken::Error for invalid tokenizer JSON" do
    expect { Gigatoken::Tokenizer.from_json("not json") }.to raise_error(Gigatoken::Error)
  end
end
