# frozen_string_literal: true

require_relative "../spec_helper"
require "fileutils"
require "tmpdir"

RSpec.describe Gigatoken::Tokenizer do
  fixture_path = File.expand_path("../../tests/fixtures/gpt2_tokenizer.json", __dir__)
  fixture = File.binread(fixture_path)
  ranks_path = File.expand_path("../fixtures/ranks.tiktoken", __dir__)

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

  it "reads special tokens from the JSON bytes whatever the String's encoding tag" do
    from_binary = described_class.from_json(fixture).special_tokens
    %w[UTF-8 US-ASCII ISO-8859-1].each do |tag|
      expect(described_class.from_json(fixture.dup.force_encoding(tag)).special_tokens).to eq(from_binary)
    end
  end

  describe ".from_tiktoken" do
    it "requires a pretokenizer keyword" do
      expect { described_class.from_tiktoken(ranks_path) }.to raise_error(ArgumentError)
    end

    it "resolves each shipped pretokenizer scheme" do
      %w[gpt2 gpt4 qwen2 qwen35 olmo3 deepseek_v3 o200k nemotron kimi].each do |scheme|
        expect(described_class.from_tiktoken(ranks_path, pretokenizer: scheme)).to be_a(described_class)
      end
    end

    it "raises Gigatoken::Error naming the scheme and the valid ones for an unknown pretokenizer" do
      expect { described_class.from_tiktoken(ranks_path, pretokenizer: "not_a_scheme") }
        .to raise_error(Gigatoken::Error, /not_a_scheme/)
    end

    it "reports caller-supplied special tokens and round-trips them through encode/decode" do
      tokenizer = described_class.from_tiktoken(ranks_path, pretokenizer: "gpt2", special_tokens: {"<|endoftext|>" => 300})
      expect(tokenizer.special_tokens).to eq({"<|endoftext|>" => 300})
      expect(tokenizer.encode("<|endoftext|>")).to eq([300])
      expect(tokenizer.decode([300]).force_encoding(Encoding::UTF_8)).to eq("<|endoftext|>")
    end
  end

  describe ".from_encoding" do
    it "resolves each packaged encoding by name with its full vocab_size" do
      {"r50k_base" => 50257, "cl100k_base" => 100277, "o200k_base" => 200019}.each do |name, vocab_size|
        expect(described_class.from_encoding(name).vocab_size).to eq(vocab_size)
      end
    end

    it "reproduces pinned token ids for each packaged encoding" do
      {
        "r50k_base" => {"hello world" => [31373, 995], "日本語 tokens" => [33768, 98, 17312, 105, 45739, 252, 16326]},
        "cl100k_base" => {"hello world" => [15339, 1917], "日本語 tokens" => [9080, 22656, 45918, 252, 11460]},
        "o200k_base" => {"hello world" => [24912, 2375], "日本語 tokens" => [9048, 40909, 20290]}
      }.each do |name, cases|
        tokenizer = described_class.from_encoding(name)
        cases.each { |text, ids| expect(tokenizer.encode(text)).to eq(ids) }
      end
    end

    it "raises Gigatoken::Error naming the bad input and the packaged encodings" do
      expect { described_class.from_encoding("not_an_encoding") }.to raise_error(Gigatoken::Error) do |error|
        expect(error.message).to include("not_an_encoding", "r50k_base", "cl100k_base", "o200k_base")
      end
    end

    it "explains p50k_base's non-dense ranks rather than only that it is unpackaged" do
      expect { described_class.from_encoding("p50k_base") }.to raise_error(Gigatoken::Error, /dense/i)
    end

    it "explains p50k_edit's non-dense ranks rather than only that it is unpackaged" do
      expect { described_class.from_encoding("p50k_edit") }.to raise_error(Gigatoken::Error, /dense/i)
    end
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

    it "dispatches a .tiktoken path to from_tiktoken, given a pretokenizer" do
      tokenizer = described_class.load(ranks_path, pretokenizer: "gpt2", special_tokens: {"<|endoftext|>" => 256})
      expect(tokenizer.vocab_size).to eq(257) # 256 bytes + <|endoftext|>
    end

    it "raises Gigatoken::Error for a .tiktoken path with no pretokenizer, naming the valid schemes" do
      expect { described_class.load(ranks_path) }.to raise_error(Gigatoken::Error, /pretokenizer/)
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

    it "dispatches a repo-id-shaped string to from_hub with a default Hub, honoring HF_ENDPOINT" do
      Dir.mktmpdir do |cache_dir|
        original = ENV.values_at("HF_HOME", "HF_ENDPOINT")
        ENV["HF_HOME"] = cache_dir
        app = ->(_request) { Protocol::HTTP::Response[200, {"x-repo-commit" => "c" * 40}, [fixture]] }

        run_hub_server(app) do |base_url|
          ENV["HF_ENDPOINT"] = base_url
          expect(described_class.load("acme/gpt2").encode("Hello, world!")).to eq([15496, 11, 995, 0])
        end
      ensure
        ENV["HF_HOME"], ENV["HF_ENDPOINT"] = original
      end
    end

    it "raises Gigatoken::Error for garbage input" do
      expect { described_class.load("../not a real path/nor a repo id") }.to raise_error(Gigatoken::Error)
    end

    it "dispatches each packaged encoding name to from_encoding, including the full special-token table" do
      {
        "r50k_base" => [50257, {"<|endoftext|>" => 50256}],
        "cl100k_base" => [100277, {
          "<|endoftext|>" => 100257,
          "<|fim_prefix|>" => 100258,
          "<|fim_middle|>" => 100259,
          "<|fim_suffix|>" => 100260,
          "<|endofprompt|>" => 100276
        }],
        "o200k_base" => [200019, {"<|endoftext|>" => 199999, "<|endofprompt|>" => 200018}]
      }.each do |name, (vocab_size, special_tokens)|
        tokenizer = described_class.load(name)
        expect(tokenizer.vocab_size).to eq(vocab_size)
        expect(tokenizer.special_tokens).to eq(special_tokens)
      end
    end

    it "still dispatches a bare legacy repo id like gpt2 to the Hub rather than the packaged registry" do
      hub_reached = false
      hub = Object.new
      hub.define_singleton_method(:hub_file) do |*|
        hub_reached = true
        raise "HUB_REACHED"
      end

      expect { described_class.load("gpt2", hub: hub) }.to raise_error("HUB_REACHED")
      expect(hub_reached).to be(true)
    end

    it "explains p50k_base's non-dense ranks rather than reaching the Hub" do
      hub_reached = false
      hub = Object.new
      hub.define_singleton_method(:hub_file) do |*|
        hub_reached = true
        raise "HUB_REACHED"
      end

      expect { described_class.load("p50k_base", hub: hub) }.to raise_error(Gigatoken::Error, /dense/i)
      expect(hub_reached).to be(false)
    end

    it "explains p50k_edit's non-dense ranks rather than reaching the Hub" do
      hub_reached = false
      hub = Object.new
      hub.define_singleton_method(:hub_file) do |*|
        hub_reached = true
        raise "HUB_REACHED"
      end

      expect { described_class.load("p50k_edit", hub: hub) }.to raise_error(Gigatoken::Error, /dense/i)
      expect(hub_reached).to be(false)
    end

    it "resolves o200k_harmony through both entry points with a read-only HF_HOME and no Hub calls, vendoring no new file" do
      Dir.mktmpdir do |dir|
        ro_home = File.join(dir, "ro")
        Dir.mkdir(ro_home)
        File.chmod(0o555, ro_home)
        original_home = ENV["HF_HOME"]
        ENV["HF_HOME"] = ro_home

        hub = Object.new
        hub.define_singleton_method(:hub_file) { |*| raise "NETWORK REACHED" }

        expect(described_class.from_encoding("o200k_harmony").vocab_size).to eq(201088)
        expect(described_class.load("o200k_harmony", hub: hub).vocab_size).to eq(201088)
      ensure
        ENV["HF_HOME"] = original_home
      end
    end

    it "resolves every packaged encoding through both entry points with a read-only HF_HOME and no Hub calls" do
      Dir.mktmpdir do |dir|
        ro_home = File.join(dir, "ro")
        Dir.mkdir(ro_home)
        File.chmod(0o555, ro_home)
        original_home = ENV["HF_HOME"]
        ENV["HF_HOME"] = ro_home

        hub = Object.new
        hub.define_singleton_method(:hub_file) { |*| raise "NETWORK REACHED" }

        %w[r50k_base cl100k_base o200k_base].each do |name|
          expect(described_class.from_encoding(name)).to be_a(described_class)
          expect(described_class.load(name, hub: hub)).to be_a(described_class)
        end
      ensure
        ENV["HF_HOME"] = original_home
      end
    end
  end

  describe "zero-copy input marshal" do
    # Ruby 4.0.6's Variable Width Allocation embeds strings up to several
    # hundred bytes (verified live: 512 B embeds, 1024 B doesn't) — well past
    # strings this short, forcing the embedded (always-copy) path.
    let(:embedded) { "hi" }
    # Comfortably over the embed threshold, so it's heap-allocated and
    # eligible for the zero-copy borrow path. `* n` always returns a fresh,
    # unfrozen string regardless of this file's frozen_string_literal magic
    # comment.
    let(:heap) { "gigatoken zero copy borrow input " * 200 }
    let(:frozen) { ("gigatoken zero copy borrow frozen input " * 100).freeze }

    it "encodes a batch matrix of embedded, heap, frozen, duplicate, shared-substring, and to_str inputs identically to per-doc encode" do
      to_str_doc = Class.new do
        def to_str
          "convert me please"
        end
      end.new
      substring = heap[100, 1200]
      texts = [embedded, heap, frozen, heap, substring, to_str_doc]
      expected = texts.map { |t| tokenizer.encode(t) }

      expect(tokenizer.encode_batch(texts)).to eq(expected)

      packed = tokenizer.encode_batch(texts, packed: true)
      expect(packed.to_a).to eq(expected)
    end

    it "leaves a heap input byte-identical, unfrozen, and immediately mutable after a ragged batch" do
      doc = heap
      original = doc.dup

      tokenizer.encode_batch([doc])

      expect(doc).to eq(original)
      expect(doc).not_to be_frozen
      expect { doc << "!" }.not_to raise_error
    end

    it "leaves a heap input byte-identical, unfrozen, and immediately mutable after a packed batch" do
      doc = heap
      original = doc.dup

      tokenizer.encode_batch([doc], packed: true)

      expect(doc).to eq(original)
      expect(doc).not_to be_frozen
      expect { doc << "!" }.not_to raise_error
    end

    it "raises TypeError for a non-string element and leaves an earlier heap string mutable" do
      doc = heap

      expect { tokenizer.encode_batch([doc, 42]) }.to raise_error(TypeError)
      expect { doc << "!" }.not_to raise_error
    end

    it "encodes to_str-convertible objects positioned before and after a heap string identically to per-doc encode, ragged and packed" do
      convert = lambda do |text|
        obj = Object.new
        obj.define_singleton_method(:to_str) { text }
        obj
      end
      texts = [convert.call("before the heap string"), heap, convert.call("after the heap string")]
      expected = texts.map { |t| tokenizer.encode(t) }

      expect(tokenizer.encode_batch(texts)).to eq(expected)

      packed = tokenizer.encode_batch(texts, packed: true)
      expect(packed.to_a).to eq(expected)
    end

    it "encodes the array as passed at entry when a to_str mutates the caller's array mid-marshal, leaving the caller's mutations visible and the earlier heap string immediately mutable after" do
      long1 = heap
      long2 = "gigatoken zero copy borrow input, second document " * 200
      docs = nil
      mutator = Object.new
      mutator.define_singleton_method(:to_str) do
        docs.pop
        docs[0] = "swapped"
        "mutant doc"
      end
      expected = [tokenizer.encode(long1), tokenizer.encode("mutant doc"), tokenizer.encode(long2)]
      docs = [long1, mutator, long2]

      expect(tokenizer.encode_batch(docs)).to eq(expected)

      expect(docs.length).to eq(2)
      expect(docs[0]).to eq("swapped")
      expect { long1 << "!" }.not_to raise_error
    end

    it "handles a to_str that freezes a later heap string mid-marshal without raising, and encodes it correctly" do
      target = "gigatoken zero copy borrow input, frozen mid-marshal " * 200
      freezer = Object.new
      freezer.define_singleton_method(:to_str) do
        target.freeze
        "frozen mid-marshal"
      end
      expected = [tokenizer.encode(target), tokenizer.encode("frozen mid-marshal")]

      result = nil
      expect { result = tokenizer.encode_batch([target, freezer]) }.not_to raise_error
      expect(result).to eq(expected)
    end
  end
end
