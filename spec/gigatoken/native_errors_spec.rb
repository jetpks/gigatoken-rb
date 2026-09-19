# frozen_string_literal: true

require_relative "../spec_helper"
require "tmpdir"

# Two classes of untrusted input used to reach a Rust `panic!` — an index past
# the vocabulary in `decode`, and a `.tiktoken` rank file whose merges cannot be
# reconstructed (I01 native F4, F5). A panic across the boundary is a Ruby
# *fatal*, which no `rescue` can catch and which would take the rest of this
# file's process down with it, so the malformed-rank-file examples come last:
# a regression there fails the run instead of truncating it.
RSpec.describe "errors from the native extension" do
  fixtures = File.expand_path("../fixtures", __dir__)
  b64 = ->(bytes) { [bytes].pack("m0") }
  all_bytes = (0..255).map { |b| "#{b64[b.chr]} #{b}" }.join("\n")

  describe "#decode of an id outside the vocabulary" do
    {
      "BPE" => File.expand_path("../../tests/fixtures/gpt2_tokenizer.json", __dir__),
      "SentencePiece" => File.expand_path("../fixtures/sp_tokenizer.json", __dir__)
    }.each do |backend, path|
      it "raises InputError naming the id on the #{backend} backend" do
        tokenizer = Gigatoken::Tokenizer.from_file(path)
        out_of_range = tokenizer.vocab_size

        expect { tokenizer.decode([out_of_range]) }
          .to raise_error(Gigatoken::InputError, /#{out_of_range}/)
      end

      it "still decodes every id inside the vocabulary on the #{backend} backend" do
        tokenizer = Gigatoken::Tokenizer.from_file(path)

        expect(tokenizer.decode([0, tokenizer.vocab_size - 1])).to be_a(String)
      end
    end
  end

  describe ".from_tiktoken on a malformed rank file" do
    # Every shape an untrusted .tiktoken can take. The first two are the ones
    # that panicked; the rest already raised (the loader's own base64, dense-
    # rank and single-byte-vocab checks), and are here so they stay errors.
    {
      "a multi-byte token with no single-byte entries" => "#{b64["ab"]} 0\n",
      "a token that does not reduce to two existing tokens" => "#{all_bytes}\n#{b64["abc"]} 256\n",
      "a token that is not base64" => "not base64! 0\n",
      "non-dense ranks" => "#{all_bytes}\n#{b64["ab"]} 999\n",
      "a duplicate rank" => "#{all_bytes}\n#{b64["ab"]} 255\n",
      "a negative rank" => "#{b64["a"]} -1\n",
      "a rank past u32" => "#{b64["a"]} 4294967296\n",
      "a line with no rank field" => "#{b64["a"]}\n",
      "an empty file" => ""
    }.each do |name, body|
      it "raises ModelError for #{name}" do
        Dir.mktmpdir do |dir|
          path = File.join(dir, "ranks.tiktoken")
          File.write(path, body)

          expect { Gigatoken::Tokenizer.from_tiktoken(path, pretokenizer: "gpt2") }
            .to raise_error(Gigatoken::ModelError)
        end
      end
    end

    it "still loads the well-formed fixture" do
      tokenizer = Gigatoken::Tokenizer.from_tiktoken(File.join(fixtures, "ranks.tiktoken"), pretokenizer: "gpt2")

      expect(tokenizer.vocab_size).to be > 0
    end
  end
end
