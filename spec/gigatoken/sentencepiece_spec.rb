# frozen_string_literal: true

require_relative "../spec_helper"
require "tmpdir"

RSpec.describe Gigatoken::Tokenizer do
  # A small byte-fallback BPE tokenizer.json (Llama-2-style Prepend+Replace
  # normalizer, no pre-tokenizer), built with the real `tokenizers` Python
  # library — see the builder report for exact provenance/derivation.
  fixture_path = File.expand_path("../fixtures/sp_tokenizer.json", __dir__)
  fixture = File.binread(fixture_path)

  let(:tokenizer) { described_class.from_file(fixture_path) }

  it "dispatches byte_fallback tokenizer.json to a SentencePieceTokenizer" do
    expect(tokenizer.instance_variable_get(:@native)).to be_a(Gigatoken::Native::SentencePieceTokenizer)
  end

  it "derives special_tokens from added_tokens" do
    expect(tokenizer.special_tokens).to eq({"<unk>" => 0, "<s>" => 1, "</s>" => 2})
  end

  describe "#encode" do
    it "encodes known vectors through trained merges" do
      expect(tokenizer.encode("hello world")).to eq([271, 276])
      expect(tokenizer.encode("hello")).to eq([271])
      expect(tokenizer.encode("world")).to eq([276])
      expect(tokenizer.encode("")).to eq([])
    end

    it "falls back to per-byte tokens for text outside the trained vocab" do
      ids = tokenizer.encode("\u{1F389}") # a single emoji, byte_fallback exercised
      expect(ids).to eq([259, 243, 162, 145, 140])
    end

    it "raises Gigatoken::Error for invalid UTF-8 input" do
      invalid = "hello \xFF".dup.force_encoding(Encoding::ASCII_8BIT)
      expect { tokenizer.encode(invalid) }.to raise_error(Gigatoken::Error, /UTF-8/)
    end
  end

  describe "#decode" do
    it "round-trips known text, and byte-fallback text, through encode/decode" do
      ["hello world", "hello", "world", "\u{1F389}", ""].each do |text|
        ids = tokenizer.encode(text)
        expect(tokenizer.decode(ids).force_encoding("UTF-8")).to eq(text)
      end
    end

    it "decodes to a BINARY-encoded String" do
      expect(tokenizer.decode([271]).encoding).to eq(Encoding::ASCII_8BIT)
    end
  end

  describe "#encode_batch" do
    let(:texts) { ["hello world", "hello", "world", "", "\u{1F389}"] }

    it "matches per-string encode" do
      expect(tokenizer.encode_batch(texts)).to eq(texts.map { |t| tokenizer.encode(t) })
    end

    it "encodes a packed batch identically to the ragged batch" do
      ragged = tokenizer.encode_batch(texts)
      packed = tokenizer.encode_batch(texts, packed: true)

      expect(packed).to be_a(Gigatoken::PackedResult)
      expect(packed.to_a).to eq(ragged)
      expect(packed.token_count).to eq(ragged.sum(&:size))
    end

    it "raises Gigatoken::Error for invalid UTF-8 in any document" do
      invalid = "world \xFF".dup.force_encoding(Encoding::ASCII_8BIT)
      expect { tokenizer.encode_batch(["hello", invalid]) }.to raise_error(Gigatoken::Error, /UTF-8/)
    end
  end

  describe "#encode_files" do
    let(:text_docs) { ["hello world", "hello", "world"] }

    it "matches encode_batch for a separator-delimited text file" do
      Dir.mktmpdir do |dir|
        path = File.join(dir, "docs.txt")
        File.write(path, text_docs.join("<|sep|>"))

        source = Gigatoken::Native::TextFileSource.new([path], separator: "<|sep|>")
        expect(tokenizer.encode_files(source)).to eq(tokenizer.encode_batch(text_docs))
      end
    end

    it "gives identical rows for parallel: false and parallel: true" do
      Dir.mktmpdir do |dir|
        path = File.join(dir, "docs.txt")
        File.write(path, text_docs.join("<|sep|>"))
        source = Gigatoken::Native::TextFileSource.new([path], separator: "<|sep|>")

        expect(tokenizer.encode_files(source, parallel: false)).to eq(tokenizer.encode_files(source, parallel: true))
      end
    end

    it "matches encode_batch for a packed result" do
      Dir.mktmpdir do |dir|
        path = File.join(dir, "docs.txt")
        File.write(path, text_docs.join("<|sep|>"))
        source = Gigatoken::Native::TextFileSource.new([path], separator: "<|sep|>")

        ragged = tokenizer.encode_batch(text_docs)
        packed = tokenizer.encode_files(source, packed: true)

        expect(packed).to be_a(Gigatoken::PackedResult)
        expect(packed.to_a).to eq(ragged)
      end
    end

    it "raises Gigatoken::Error for a non-UTF-8 separator" do
      Dir.mktmpdir do |dir|
        path = File.join(dir, "docs.txt")
        File.write(path, "hello world")
        separator = "\xA9".dup.force_encoding(Encoding::ASCII_8BIT)
        source = Gigatoken::Native::TextFileSource.new([path], separator: separator)

        expect { tokenizer.encode_files(source) }.to raise_error(Gigatoken::Error, /separator/)
      end
    end

    it "raises Gigatoken::Error for invalid UTF-8 file contents" do
      Dir.mktmpdir do |dir|
        path = File.join(dir, "docs.txt")
        File.binwrite(path, "hello \xFF world")
        source = Gigatoken::Native::TextFileSource.new([path])

        expect { tokenizer.encode_files(source) }.to raise_error(Gigatoken::Error, /UTF-8/)
      end
    end
  end

  describe "vocabulary" do
    it "reports the vocab size" do
      expect(tokenizer.vocab_size).to eq(277)
    end

    it "maps ids to their token bytes, including byte-fallback and merged tokens" do
      vocab = tokenizer.vocab
      expect(vocab[3]).to eq("\x00".b)
      expect(vocab[259]).to eq("▁".b)
      expect(vocab[270]).to eq("hello".b)
      expect(vocab[275]).to eq("world".b)
      expect(vocab[276]).to eq("▁world".b)
    end

    it "reports the merge rules in rank order" do
      expected = [
        %w[h e], %w[he l], %w[hel l], %w[hell o], ["▁", "hello"],
        %w[w o], %w[wo r], %w[wor l], %w[worl d], ["▁", "world"]
      ].map { |a, b| [a.b, b.b] }
      expect(tokenizer.merges).to eq(expected)
    end
  end

  it "loads identically via Tokenizer.from_json" do
    from_json = Gigatoken::Tokenizer.from_json(fixture)
    expect(from_json.encode("hello world")).to eq(tokenizer.encode("hello world"))
  end
end
