# frozen_string_literal: true

require_relative "../spec_helper"

RSpec.describe Gigatoken::Encodings do
  describe "::NAMES" do
    it "includes every packaged tiktoken encoding" do
      expect(described_class::NAMES).to include("r50k_base", "cl100k_base", "o200k_base", "o200k_harmony")
    end

    it "excludes names known but not packaged, each carrying an unpackable reason" do
      expect(described_class::NAMES).not_to include("p50k_base", "p50k_edit")
      expect(described_class.unpackable_reason("p50k_base")).to match(/dense/i)
      expect(described_class.unpackable_reason("p50k_edit")).to match(/dense/i)
    end
  end

  describe ".[]" do
    it "resolves r50k_base to its vendored rank file, scheme, and special tokens" do
      encoding = described_class["r50k_base"]
      expect(File.basename(encoding[:rank_file])).to eq("r50k_base.tiktoken")
      expect(encoding[:pretokenizer]).to eq("gpt2")
      expect(encoding[:special_tokens]).to eq({"<|endoftext|>" => 50256})
    end

    it "resolves cl100k_base to its vendored rank file, scheme, and special tokens" do
      encoding = described_class["cl100k_base"]
      expect(File.basename(encoding[:rank_file])).to eq("cl100k_base.tiktoken")
      expect(encoding[:pretokenizer]).to eq("gpt4")
      expect(encoding[:special_tokens]).to eq({
        "<|endoftext|>" => 100257,
        "<|fim_prefix|>" => 100258,
        "<|fim_middle|>" => 100259,
        "<|fim_suffix|>" => 100260,
        "<|endofprompt|>" => 100276
      })
    end

    it "resolves o200k_base to its vendored rank file, scheme, and special tokens" do
      encoding = described_class["o200k_base"]
      expect(File.basename(encoding[:rank_file])).to eq("o200k_base.tiktoken")
      expect(encoding[:pretokenizer]).to eq("o200k")
      expect(encoding[:special_tokens]).to eq({"<|endoftext|>" => 199999, "<|endofprompt|>" => 200018})
    end

    it "resolves o200k_harmony to o200k_base's vendored rank file and scheme, with its own special-token table" do
      encoding = described_class["o200k_harmony"]
      expect(encoding[:rank_file]).to eq(described_class["o200k_base"][:rank_file])
      expect(encoding[:pretokenizer]).to eq("o200k")

      special = encoding[:special_tokens]
      expect(special.size).to eq(1091)
      expect(special.values_at(
        "<|startoftext|>", "<|endoftext|>", "<|return|>", "<|constrain|>", "<|channel|>",
        "<|start|>", "<|end|>", "<|message|>", "<|call|>", "<|endofprompt|>"
      )).to eq([199998, 199999, 200002, 200003, 200005, 200006, 200007, 200008, 200012, 200018])
      expect(special["<|reserved_200018|>"]).to eq(200018)
      expect(special).not_to have_key("<|reserved_200002|>")
      expect(special.keys.count { |k| k.start_with?("<|reserved_") }).to eq(1081)
    end

    it "returns nil for an unpackaged name" do
      expect(described_class["not_an_encoding"]).to be_nil
    end

    it "resolves a Symbol name the same as the String" do
      expect(described_class[:cl100k_base]).to equal(described_class["cl100k_base"])
    end

    # Tokenizer#special_tokens hands the registry's own Hash back, so a
    # shallow freeze would let one caller's poke rewrite every later load.
    it "is frozen all the way down: entry, special tokens, and rank file" do
      described_class::NAMES.each do |name|
        encoding = described_class[name]

        expect(encoding).to be_frozen
        expect(encoding[:rank_file]).to be_frozen
        expect(encoding[:special_tokens]).to be_frozen
        expect { encoding[:special_tokens]["<|pwned|>"] = 1 }.to raise_error(FrozenError)
        expect { encoding[:pretokenizer] = "gpt2" }.to raise_error(FrozenError)
      end
    end

    it "points each rank file at a file that actually exists on disk" do
      described_class::NAMES.each do |name|
        expect(File.exist?(described_class[name][:rank_file])).to be(true)
      end
    end
  end

  describe ".unpackable_reason" do
    it "explains why p50k_base cannot be packaged" do
      expect(described_class.unpackable_reason("p50k_base")).to match(/dense/i)
    end

    it "explains why p50k_edit cannot be packaged" do
      expect(described_class.unpackable_reason("p50k_edit")).to match(/dense/i)
    end

    it "returns nil for a packaged name" do
      expect(described_class.unpackable_reason("cl100k_base")).to be_nil
    end

    it "returns nil for a name with no reason on record" do
      expect(described_class.unpackable_reason("not_an_encoding")).to be_nil
    end

    it "explains a Symbol name too" do
      expect(described_class.unpackable_reason(:p50k_base)).to match(/dense/i)
    end

    # The registry only records the reason; Tokenizer.from_encoding is what
    # raises it, and a caller asking for an unloadable model deserves the
    # class that says so.
    it "reaches the caller as a Gigatoken::ModelError through Tokenizer.from_encoding" do
      expect { Gigatoken::Tokenizer.from_encoding("p50k_base") }
        .to raise_error(Gigatoken::ModelError, /#{Regexp.escape(described_class.unpackable_reason("p50k_base"))}/)
    end
  end
end
