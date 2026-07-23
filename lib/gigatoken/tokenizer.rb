# frozen_string_literal: true

require "json"

module Gigatoken
  # A BPE tokenizer: encode, batch encode, decode, and vocabulary
  # introspection over the native `Gigatoken::Native::BPETokenizer`.
  class Tokenizer
    TIKTOKEN_ENDOFTEXT = "<|endoftext|>"
    private_constant :TIKTOKEN_ENDOFTEXT

    # Load from in-memory tokenizer.json contents (String or bytes).
    def self.from_json(data)
      native = Native::BPETokenizer.from_hf_json(data)
      new(native, special_tokens: special_tokens_from_json(data))
    end

    # Load from a tokenizer.json path, or a directory containing one.
    def self.from_file(path)
      path = File.join(path, "tokenizer.json") if File.directory?(path)
      from_json(File.binread(path))
    end

    # Load from a .tiktoken mergeable-ranks file.
    def self.from_tiktoken(path)
      native = Native::BPETokenizer.from_tiktoken(path.to_s)
      new(native, special_tokens: {TIKTOKEN_ENDOFTEXT => native.vocab_size - 1})
    end

    # Load tokenizer.json from HuggingFace Hub repo `repo_id` at `revision`
    # (downloaded directly; huggingface_hub is not required).
    def self.from_hub(repo_id, revision: "main", hub: Hub.new)
      from_file(hub.hub_file(repo_id, "tokenizer.json", revision: revision))
    end

    # Load from any of the supported source shapes: an existing file or
    # directory path (a tokenizer.json, or a directory containing one), a
    # .tiktoken vocabulary file, or a HuggingFace Hub repo id like
    # "openai-community/gpt2".
    def self.load(source, revision: "main", hub: Hub.new)
      source = source.to_s
      return from_tiktoken(source) if source.end_with?(".tiktoken")
      return from_file(source) if File.exist?(source)
      return from_hub(source, revision: revision, hub: hub) if Hub.looks_like_repo_id?(source)

      raise Error, "#{source.inspect}: no such file or directory, not a .tiktoken path, and doesn't look like a HuggingFace Hub repo id"
    end

    def self.special_tokens_from_json(data)
      added = JSON.parse(data.dup.force_encoding(Encoding::UTF_8))["added_tokens"] || []
      added.each_with_object({}) { |t, h| h[t["content"]] = t["id"] if t["special"] }
    end
    private_class_method :special_tokens_from_json

    def initialize(native, special_tokens: {})
      @native = native
      @special_tokens = special_tokens
    end

    def encode(text)
      @native.encode(text)
    end

    def encode_batch(texts)
      @native.encode_batch(texts)
    end

    def decode(ids)
      @native.decode(ids)
    end

    def vocab_size
      @native.vocab_size
    end

    def vocab
      @native.vocab
    end

    def merges
      @native.merges
    end

    attr_reader :special_tokens
  end
end
