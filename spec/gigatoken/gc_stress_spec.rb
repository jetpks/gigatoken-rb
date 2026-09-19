# frozen_string_literal: true

require_relative "../spec_helper"

# Every native path with a GC at each allocation: a Ruby object the
# extension hands out, or holds without marking, would be freed or moved
# mid-call and read afterwards. Small inputs and the tiny fixture
# tokenizers keep this to seconds. The stress is on only inside each
# example, and only a minor GC per allocation (flag 0x01), which is what
# catches an unmarked young object; the whole suite under it takes an
# hour-plus on Ruby 3.4.
RSpec.describe "native objects under GC stress" do
  fixtures = File.expand_path("../fixtures", __dir__)
  docs = ["The quick brown fox jumps over the lazy dog.", +"unfrozen " * 200, "日本語のテキスト", "x" * 2000, ""]

  def stressed
    GC.stress = 1
    yield
  ensure
    GC.stress = false
  end

  let(:bpe) do
    Gigatoken::Tokenizer.from_tiktoken(File.join(fixtures, "ranks.tiktoken"), pretokenizer: "gpt2",
      special_tokens: {"<|endoftext|>" => 256})
  end
  let(:sp) { Gigatoken::Tokenizer.from_json(File.binread(File.join(fixtures, "sp_tokenizer.json"))) }

  it "encodes, batches, packs and decodes on the BPE path" do
    tok = bpe
    expected = docs.map { |d| tok.encode(d) }
    stressed do
      expect(docs.map { |d| tok.encode(d) }).to eq(expected)
      expect(tok.encode_batch(docs)).to eq(expected)
      packed = tok.encode_batch(docs, packed: true)
      expect(packed.to_a).to eq(expected)
      expect(packed[1]).to eq(expected[1])
      expect(tok.decode(expected[0])).to eq(docs[0].b)
      expect(tok.vocab_size).to eq(257)
    end
  end

  it "reads and encodes files in every format" do
    tok = bpe
    text = File.join(fixtures, "docs.txt")
    jsonl = Gigatoken::Native::JsonlFileSource.new([File.join(fixtures, "docs.jsonl")])
    parquet = Gigatoken::Native::ParquetFileSource.new([File.join(fixtures, "docs.parquet")])
    from_text = tok.encode_files(text, separator: "<|endoftext|>")
    from_jsonl = tok.encode_files(jsonl)
    from_parquet = tok.encode_files(parquet)
    stressed do
      expect(tok.encode_files(text, separator: "<|endoftext|>")).to eq(from_text)
      expect(tok.encode_files(text, separator: "<|endoftext|>", packed: true).to_a).to eq(from_text)
      expect(tok.encode_files(text, separator: "<|endoftext|>", parallel: false)).to eq(from_text)
      expect(tok.encode_files(jsonl)).to eq(from_jsonl)
      expect(tok.encode_files(parquet, packed: true).to_a).to eq(from_parquet)
    end
  end

  it "encodes, batches and packs on the SentencePiece path" do
    tok = sp
    inputs = ["hello world", "abc", ""]
    expected = inputs.map { |d| tok.encode(d) }
    stressed do
      expect(inputs.map { |d| tok.encode(d) }).to eq(expected)
      expect(tok.encode_batch(inputs)).to eq(expected)
      expect(tok.encode_batch(inputs, packed: true).to_a).to eq(expected)
      expect(tok.decode(expected[0])).to be_a(String)
    end
  end
end
