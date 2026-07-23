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

    it "accepts a bare path with no separator, through the facade" do
      whole = File.binread(File.join(fixtures, "docs.txt"))
      rows = tokenizer.encode_files(File.join(fixtures, "docs.txt"))
      expect(rows).to eq([tokenizer.encode(whole)])
    end

    it "accepts a bare path with an explicit nil separator, same as omitted" do
      omitted = tokenizer.encode_files(File.join(fixtures, "docs.txt"))
      explicit_nil = tokenizer.encode_files([File.join(fixtures, "docs.txt")], separator: nil)
      expect(explicit_nil).to eq(omitted)
      expect(explicit_nil.length).to eq(1)
    end

    it "matches the Native source rows for an array of bare paths plus separator" do
      source = Gigatoken::Native::TextFileSource.new([File.join(fixtures, "docs.txt")], separator: "<|endoftext|>")
      rows = tokenizer.encode_files([File.join(fixtures, "docs.txt")], separator: "<|endoftext|>")
      expect(rows).to eq(tokenizer.encode_files(source))
    end

    it "treats parallel: nil the same as omitting parallel:" do
      omitted = tokenizer.encode_files(File.join(fixtures, "docs.txt"))
      explicit_nil = tokenizer.encode_files(File.join(fixtures, "docs.txt"), parallel: nil)
      expect(explicit_nil).to eq(omitted)
    end

    it "gives identical rows for parallel: false and parallel: true with a text file" do
      source = Gigatoken::Native::TextFileSource.new([File.join(fixtures, "docs.txt")], separator: "<|endoftext|>")
      expect(tokenizer.encode_files(source, parallel: false)).to eq(tokenizer.encode_files(source, parallel: true))
    end

    it "gives identical rows for parallel: false and parallel: true with a jsonl file" do
      source = Gigatoken::Native::JsonlFileSource.new([File.join(fixtures, "docs.jsonl")])
      expect(tokenizer.encode_files(source, parallel: false)).to eq(tokenizer.encode_files(source, parallel: true))
    end

    it "gives identical rows for parallel: false and parallel: true with a parquet file" do
      source = Gigatoken::Native::ParquetFileSource.new([File.join(fixtures, "docs.parquet")])
      expect(tokenizer.encode_files(source, parallel: false)).to eq(tokenizer.encode_files(source, parallel: true))
    end

    it "matches encode_batch for a packed result" do
      source = Gigatoken::Native::TextFileSource.new([File.join(fixtures, "docs.txt")], separator: "<|endoftext|>")
      ragged = tokenizer.encode_batch(text_docs)
      packed = tokenizer.encode_files(source, packed: true)

      expect(packed).to be_a(Gigatoken::PackedResult)
      expect(packed.to_a).to eq(ragged)
      expect(packed.token_count).to eq(ragged.sum(&:size))
    end

    it "gives identical packed rows for parallel: false and parallel: true" do
      source = Gigatoken::Native::TextFileSource.new([File.join(fixtures, "docs.txt")], separator: "<|endoftext|>")
      serial = tokenizer.encode_files(source, packed: true, parallel: false)
      parallel = tokenizer.encode_files(source, packed: true, parallel: true)
      expect(serial.to_a).to eq(parallel.to_a)
    end

    it "treats packed: nil the same as omitting packed:" do
      source = Gigatoken::Native::TextFileSource.new([File.join(fixtures, "docs.txt")], separator: "<|endoftext|>")
      omitted = tokenizer.encode_files(source)
      explicit_nil = tokenizer.encode_files(source, packed: nil)
      expect(explicit_nil).to eq(omitted)
    end
  end

  describe "Native::JsonlFileSource" do
    it "treats field: nil the same as omitting field:" do
      omitted = Gigatoken::Native::JsonlFileSource.new([File.join(fixtures, "docs.jsonl")])
      explicit_nil = Gigatoken::Native::JsonlFileSource.new([File.join(fixtures, "docs.jsonl")], field: nil)
      expect(tokenizer.encode_files(explicit_nil)).to eq(tokenizer.encode_files(omitted))
    end
  end

  describe "Native::ParquetFileSource" do
    it "treats column: nil the same as omitting column:" do
      omitted = Gigatoken::Native::ParquetFileSource.new([File.join(fixtures, "docs.parquet")])
      explicit_nil = Gigatoken::Native::ParquetFileSource.new([File.join(fixtures, "docs.parquet")], column: nil)
      expect(tokenizer.encode_files(explicit_nil)).to eq(tokenizer.encode_files(omitted))
    end
  end
end
