# frozen_string_literal: true

require_relative "../spec_helper"

RSpec.describe Gigatoken::Tokenizer do
  fixtures = File.expand_path("../fixtures", __dir__)
  fixture_path = File.expand_path("../../tests/fixtures/gpt2_tokenizer.json", __dir__)

  let(:tokenizer) { described_class.from_file(fixture_path) }

  let(:text_docs) do
    [
      "The quick brown fox jumps over the lazy dog.",
      "Café society meets at dawn.",
      "日本語のテキストです。",
      "A final short document."
    ]
  end

  describe "#encode_files" do
    it "matches encode_batch for a separator-delimited text file" do
      source = Gigatoken::Native::TextFileSource.new([File.join(fixtures, "docs.txt")], separator: "<|endoftext|>")
      expect(tokenizer.encode_files(source)).to eq(tokenizer.encode_batch(text_docs))
    end

    it "accepts a bare path plus separator:, wrapping it in a TextFileSource" do
      rows = tokenizer.encode_files(File.join(fixtures, "docs.txt"), separator: "<|endoftext|>")
      expect(rows).to eq(tokenizer.encode_batch(text_docs))
    end

    it "accepts an array of bare paths" do
      rows = tokenizer.encode_files([File.join(fixtures, "docs.txt")], separator: "<|endoftext|>")
      expect(rows).to eq(tokenizer.encode_batch(text_docs))
    end

    it "matches encode_batch for a jsonl file" do
      source = Gigatoken::Native::JsonlFileSource.new([File.join(fixtures, "docs.jsonl")])
      expect(tokenizer.encode_files(source)).to eq(tokenizer.encode_batch(text_docs))
    end

    it "matches encode_batch for a parquet file" do
      source = Gigatoken::Native::ParquetFileSource.new([File.join(fixtures, "docs.parquet")])
      expect(tokenizer.encode_files(source)).to eq(tokenizer.encode_batch(text_docs))
    end

    it "matches its uncompressed twin for a gzipped text file" do
      plain = Gigatoken::Native::TextFileSource.new([File.join(fixtures, "docs.txt")], separator: "<|endoftext|>")
      gzipped = Gigatoken::Native::TextFileSource.new([File.join(fixtures, "docs.txt.gz")], separator: "<|endoftext|>")
      expect(tokenizer.encode_files(gzipped)).to eq(tokenizer.encode_files(plain))
    end

    it "treats a whole file with no separator as a single document" do
      source = Gigatoken::Native::TextFileSource.new([File.join(fixtures, "docs.txt")])
      whole = File.binread(File.join(fixtures, "docs.txt"))
      expect(tokenizer.encode_files(source)).to eq([tokenizer.encode(whole)])
    end

    it "raises Gigatoken::Error for a missing file" do
      source = Gigatoken::Native::TextFileSource.new(["/no/such/file.txt"])
      expect { tokenizer.encode_files(source) }.to raise_error(Gigatoken::Error)
    end
  end
end
