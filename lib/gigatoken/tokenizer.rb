# frozen_string_literal: true

require "json"

module Gigatoken
  # A tokenizer: encode, batch encode, decode, and vocabulary introspection
  # over a native `Gigatoken::Native::BPETokenizer` or
  # `Gigatoken::Native::SentencePieceTokenizer`.
  class Tokenizer
    FILE_SOURCE_CLASSES = [Native::TextFileSource, Native::JsonlFileSource, Native::ParquetFileSource].freeze
    private_constant :FILE_SOURCE_CLASSES

    # Load from in-memory tokenizer.json contents (String or bytes). Backed
    # by a BPETokenizer or a SentencePieceTokenizer, per the model's
    # byte_fallback flag.
    def self.from_json(data)
      native = Native.load_hf_json(data)
      new(native, special_tokens: special_tokens_from_json(data))
    end

    # Load from a tokenizer.json path, or a directory containing one.
    def self.from_file(path)
      path = File.join(path, "tokenizer.json") if File.directory?(path)
      from_json(File.binread(path))
    end

    # Load from a .tiktoken mergeable-ranks file. The file carries neither a
    # pretokenization scheme nor special tokens — nothing is guessed here,
    # so `pretokenizer:` is required: one of the schemes gigatoken ships
    # (see Native.pretokenizer_names, e.g. "gpt2"/"r50k", "gpt4"/"cl100k",
    # "o200k", "qwen2", "qwen35", "olmo3", "deepseek_v3", "nemotron", "kimi").
    # `special_tokens:` maps token content to id (none by default).
    def self.from_tiktoken(path, pretokenizer:, special_tokens: {})
      native = Native::BPETokenizer.from_tiktoken(path.to_s, pretokenizer, special_tokens)
      new(native, special_tokens: special_tokens)
    end

    # Load one of the tiktoken encodings gigatoken vendors ranks for, by
    # name — see Gigatoken::Encodings::NAMES — entirely from the vendored
    # files: no network, no writable cache.
    def self.from_encoding(name)
      encoding = Encodings[name]
      return from_tiktoken(encoding[:rank_file], pretokenizer: encoding[:pretokenizer], special_tokens: encoding[:special_tokens]) if encoding

      reason = Encodings.unpackable_reason(name)
      detail = reason ? " — #{reason}" : ""
      raise Error, "#{name.inspect}: not a packaged encoding#{detail} (packaged encodings: #{Encodings::NAMES.join(", ")})"
    end

    # Load tokenizer.json from HuggingFace Hub repo `repo_id` at `revision`
    # (downloaded directly; huggingface_hub is not required).
    def self.from_hub(repo_id, revision: "main", hub: Hub.new)
      from_file(hub.hub_file(repo_id, "tokenizer.json", revision: revision))
    end

    # Load from any of the supported source shapes: an existing file or
    # directory path (a tokenizer.json, or a directory containing one), a
    # .tiktoken vocabulary file, a packaged encoding name (see
    # Gigatoken::Encodings::NAMES, e.g. "cl100k_base"), or a HuggingFace Hub
    # repo id like "openai-community/gpt2". A .tiktoken file carries no
    # pretokenizer scheme of its own, so one must be named explicitly via
    # `pretokenizer:` — nothing here is guessed. Packaged encoding names are
    # checked before the Hub-repo-id shape: a bare name like "o200k_base" is
    # also shaped like a legacy repo id, and must resolve locally rather
    # than reach the network. Names the registry knows but doesn't package
    # (see Encodings.unpackable_reason, e.g. "p50k_base") are intercepted
    # here too, raising the same explanation from_encoding gives rather than
    # reaching the Hub — but only those; an unrecognized bare name like
    # "gpt2" still dispatches to the Hub.
    def self.load(source, pretokenizer: nil, special_tokens: {}, revision: "main", hub: nil)
      source = source.to_s
      if source.end_with?(".tiktoken")
        unless pretokenizer
          raise Error, "#{source.inspect}: a .tiktoken file carries no pretokenizer scheme of its own — " \
            "pass pretokenizer: (one of #{Native.pretokenizer_names.join(", ")})"
        end
        return from_tiktoken(source, pretokenizer: pretokenizer, special_tokens: special_tokens)
      end
      return from_file(source) if File.exist?(source)
      return from_encoding(source) if Encodings::NAMES.include?(source) || Encodings.unpackable_reason(source)
      return from_hub(source, revision: revision, hub: hub || Hub.new) if Hub.looks_like_repo_id?(source)

      raise Error, "#{source.inspect}: no such file or directory, not a .tiktoken path, and doesn't look like a HuggingFace Hub repo id"
    end

    def self.special_tokens_from_json(data)
      added = JSON.parse(data)["added_tokens"] || []
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

    # Returns a ragged Array of Arrays of token ids, one row per document —
    # or, with `packed: true`, a Gigatoken::PackedResult (one IO::Buffer of
    # token ids plus per-document lengths), avoiding the per-token Ruby
    # array materialization the ragged shape costs.
    def encode_batch(texts, packed: false)
      if packed
        PackedResult.new(*@native.encode_batch_packed(texts))
      else
        @native.encode_batch(texts)
      end
    end

    # Tokenize whole files in Rust: reads and encodes them in one fused pass
    # without the documents ever becoming Ruby objects. `source` is a
    # Native::{Text,Jsonl,Parquet}FileSource, a single path, or an array of
    # paths; bare path(s) are wrapped in a TextFileSource (with `separator`,
    # if given). Returns a ragged Array of Arrays of token ids, one row per
    # document — or, with `packed: true`, a Gigatoken::PackedResult. `parallel:
    # false` loads and encodes everything on the calling thread instead, with
    # identical output, never touching the core worker pool.
    def encode_files(source, separator: nil, parallel: true, packed: false)
      source = Native::TextFileSource.new(Array(source).map(&:to_s), separator: separator) unless FILE_SOURCE_CLASSES.any? { |klass| source.is_a?(klass) }
      if packed
        PackedResult.new(*@native.encode_files_packed(source, parallel: parallel))
      else
        @native.encode_files(source, parallel: parallel)
      end
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

    # Cached pretoken/unit entries on this tokenizer's single-document
    # encode path — see Gigatoken.max_cache_bytes.
    def cache_entries
      @native.cache_entries
    end

    attr_reader :special_tokens
  end
end
