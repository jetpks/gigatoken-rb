# frozen_string_literal: true

require_relative "../spec_helper"
require "tiktoken_ruby"
require "tokenizers"
require "zlib"

# Proves each packaged encoding byte-identical to an independent
# implementation over this repo's own source and docs: the rank-backed
# (tiktoken) ones to tiktoken_ruby, the JSON-backed (HuggingFace) ones to
# HuggingFace's own `tokenizers` gem. The tiktoken half is checked in both
# directions, because gigatoken always honours an encoding's special tokens
# and tiktoken's default `encode` does not (this repo's own files contain
# "<|endoftext|>" as literal text), so a one-sided comparison can't tell a
# correct encoder from one checked against the wrong oracle method.
RSpec.describe "packaged encodings against their oracles" do
  corpus_paths = (Dir["lib/**/*.rb"] + Dir["spec/**/*.rb"] + Dir["src/**/*.rs"] + ["README.md", "CHANGELOG.md"])
    .select { |path| File.file?(path) }.sort
  corpus = corpus_paths.map { |path| File.read(path, encoding: "UTF-8") }

  it "covers a non-trivial corpus of the repo's own text" do
    expect(corpus_paths.size).to be >= 60
    expect(corpus.sum(&:bytesize)).to be >= 1_000_000
  end

  # o200k_harmony is excluded here and proven separately, below, by
  # reduction plus pinning rather than against tiktoken_ruby: tiktoken_ruby
  # 0.0.17's own o200k_harmony table drops "<|endofprompt|>" (it encodes
  # the literal as six ordinary-text tokens, not [200018]) while treating
  # "<|reserved_200018|>" as the sole literal at that id — measured live
  # against this gem's tiktoken_ruby dependency. openai/tiktoken 0.9.0
  # keeps both keys at id 200018, because Python dict construction
  # preserves both; openai/tiktoken is authoritative here, and
  # tiktoken_ruby is the outlier (see PROVENANCE.md). Comparing harmony
  # against tiktoken_ruby would fail on tiktoken_ruby's defect, not
  # gigatoken's — do not restore that comparison to "fix" this.
  (Gigatoken::Encodings::NAMES.select { |name| Gigatoken::Encodings[name][:rank_file] } - ["o200k_harmony"]).each do |name|
    describe name do
      entry = Gigatoken::Encodings[name]
      oracle = Tiktoken.get_encoding(name)

      it "is byte-identical to tiktoken_ruby's encode_with_special_tokens for the packaged tokenizer" do
        packaged = Gigatoken::Tokenizer.from_encoding(name)
        corpus.each { |text| expect(packaged.encode(text)).to eq(oracle.encode_with_special_tokens(text)) }
      end

      it "is byte-identical to tiktoken_ruby's plain encode for a special_tokens: {} tokenizer" do
        plain = Gigatoken::Tokenizer.from_tiktoken(entry[:rank_file], pretokenizer: entry[:pretokenizer], special_tokens: {})
        corpus.each { |text| expect(plain.encode(text)).to eq(oracle.encode(text)) }
      end
    end
  end

  # The JSON-backed encodings are checked against `tokenizers` (HuggingFace's
  # own implementation) loading the same decompressed file, with
  # `add_special_tokens: false`: that flag skips the post-processor, which
  # gigatoken never applies — muse_spark's TemplateProcessing would otherwise
  # prepend <|begin_of_text|> to every oracle row. The added tokens
  # themselves (this repo's own files contain literals like "<|im_start|>")
  # are matched by both sides regardless of the flag.
  Gigatoken::Encodings::NAMES.select { |name| Gigatoken::Encodings[name][:json_file] }.each do |name|
    describe name do
      it "is byte-identical to tokenizers' encode with add_special_tokens: false" do
        packaged = Gigatoken::Tokenizer.from_encoding(name)
        json = Zlib.gunzip(File.binread(Gigatoken::Encodings[name][:json_file])).force_encoding(Encoding::UTF_8)
        oracle = Tokenizers::Tokenizer.from_str(json)
        corpus.each { |text| expect(packaged.encode(text)).to eq(oracle.encode(text, add_special_tokens: false).ids) }
      end
    end
  end

  # o200k_harmony has no correct oracle to check against (see above), so it
  # is proven two other ways: by reduction to o200k_base, which already
  # carries the tiktoken_ruby proof above for the ranks and split regex
  # harmony reuses verbatim; and by pinning the special-token table harmony
  # adds on top, transcribed from openai/tiktoken 0.9.0 (PROVENANCE.md).
  describe "o200k_harmony" do
    harmony = Gigatoken::Tokenizer.from_encoding("o200k_harmony")
    base = Gigatoken::Tokenizer.from_encoding("o200k_base")
    added_literals = Gigatoken::Encodings["o200k_harmony"][:special_tokens].keys -
      Gigatoken::Encodings["o200k_base"][:special_tokens].keys

    it "reduces to o200k_base for corpus files containing none of harmony's added special literals" do
      comparable = corpus_paths.zip(corpus).reject { |_path, text| added_literals.any? { |literal| text.include?(literal) } }
      expect(comparable.size).to be >= 50

      comparable.each { |_path, text| expect(harmony.encode(text)).to eq(base.encode(text)) }
    end

    it "pins the special-token table's shape" do
      special = Gigatoken::Encodings["o200k_harmony"][:special_tokens]
      expect(special.size).to eq(1091)
      expect(special.values_at(
        "<|startoftext|>", "<|endoftext|>", "<|return|>", "<|constrain|>", "<|channel|>",
        "<|start|>", "<|end|>", "<|message|>", "<|call|>", "<|endofprompt|>"
      )).to eq([199998, 199999, 200002, 200003, 200005, 200006, 200007, 200008, 200012, 200018])
      expect(special["<|endofprompt|>"]).to eq(200018)
      expect(special["<|reserved_200018|>"]).to eq(200018)
    end

    it "round-trips a real harmony control sequence" do
      expect(harmony.encode("<|start|>system<|message|>hi<|end|><|return|>"))
        .to eq([200006, 17360, 200008, 3686, 200007, 200002])
    end
  end
end
