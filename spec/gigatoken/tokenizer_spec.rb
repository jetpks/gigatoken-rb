# frozen_string_literal: true

require_relative "../spec_helper"
require "fileutils"
require "tmpdir"

RSpec.describe Gigatoken::Tokenizer do
  fixture_path = File.expand_path("../../tests/fixtures/gpt2_tokenizer.json", __dir__)
  fixture = File.binread(fixture_path)

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

  it "encodes a packed batch identically to the ragged batch" do
    texts = ["Hello, world!", "", "café", "日本語のテキスト", "a longer sentence for batching."]
    ragged = tokenizer.encode_batch(texts)
    packed = tokenizer.encode_batch(texts, packed: true)

    expect(packed).to be_a(Gigatoken::PackedResult)
    expect(packed.to_a).to eq(ragged)
    expect(packed.token_count).to eq(ragged.sum(&:size))
  end

  it "encodes a packed batch identically to the ragged batch across many chunks" do
    # Large enough (several MB, split across many documents) to force the
    # parallel chunked gather (MIN_CHUNK_BYTES is 1 MiB) rather than the
    # single-chunk fast path, exercising the zero-copy packed gather's
    # overlapped commit end-to-end.
    texts = Array.new(300) { |i| "Document #{i}: #{"The quick brown fox jumps over the lazy dog. " * 250}" }
    ragged = tokenizer.encode_batch(texts)
    packed = tokenizer.encode_batch(texts, packed: true)

    expect(packed).to be_a(Gigatoken::PackedResult)
    expect(packed.to_a).to eq(ragged)
    expect(packed.token_count).to eq(ragged.sum(&:size))
  end

  it "treats packed: nil the same as omitting packed: on encode_batch" do
    texts = ["Hello, world!", "café"]
    expect(tokenizer.encode_batch(texts, packed: nil)).to eq(tokenizer.encode_batch(texts))
  end

  it "reports the vocab size and decodes to a BINARY-encoded String" do
    expect(tokenizer.vocab_size).to eq(50257)
    expect(tokenizer.decode([15496]).encoding).to eq(Encoding::ASCII_8BIT)
  end

  it "raises Gigatoken::Error for invalid tokenizer JSON" do
    expect { Gigatoken::Tokenizer.from_json("not json") }.to raise_error(Gigatoken::Error)
  end

  describe ".load" do
    it "dispatches an existing tokenizer.json path to from_file" do
      tokenizer = described_class.load(fixture_path)
      expect(tokenizer.encode("Hello, world!")).to eq([15496, 11, 995, 0])
    end

    it "dispatches a directory containing tokenizer.json to from_file" do
      Dir.mktmpdir do |dir|
        FileUtils.cp(fixture_path, File.join(dir, "tokenizer.json"))
        tokenizer = described_class.load(dir)
        expect(tokenizer.encode("Hello, world!")).to eq([15496, 11, 995, 0])
      end
    end

    it "dispatches a .tiktoken path to from_tiktoken" do
      Dir.mktmpdir do |dir|
        path = File.join(dir, "vocab.tiktoken")
        ranks = Array.new(256) { |byte| "#{[byte.chr].pack("m0")} #{byte}" }.join("\n")
        File.write(path, ranks)

        tokenizer = described_class.load(path)
        expect(tokenizer.vocab_size).to eq(257) # 256 bytes + <|endoftext|>
      end
    end

    it "dispatches a repo-id-shaped string to from_hub, via an injected Hub" do
      Dir.mktmpdir do |cache_dir|
        original_home = ENV["HF_HOME"]
        ENV["HF_HOME"] = cache_dir
        app = ->(_request) { Protocol::HTTP::Response[200, {"x-repo-commit" => "b" * 40}, [fixture]] }

        run_hub_server(app) do |base_url|
          hub = Gigatoken::Hub.new(endpoint: base_url)
          tokenizer = described_class.load("acme/gpt2", hub: hub)
          expect(tokenizer.encode("Hello, world!")).to eq([15496, 11, 995, 0])
        end
      ensure
        ENV["HF_HOME"] = original_home
      end
    end

    it "raises Gigatoken::Error for garbage input" do
      expect { described_class.load("../not a real path/nor a repo id") }.to raise_error(Gigatoken::Error)
    end
  end
end
