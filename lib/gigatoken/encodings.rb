# frozen_string_literal: true

module Gigatoken
  # The encodings gigatoken vendors a file for. For the tiktoken ones, the
  # pieces a .tiktoken file doesn't carry: its pretokenizer scheme and
  # special-token table (see Tokenizer.from_tiktoken and lib/gigatoken/
  # encodings/PROVENANCE.md, the source of truth this is transcribed from).
  # The HuggingFace ones are whole tokenizer.json files, which carry theirs.
  module Encodings
    DATA_DIR = File.expand_path("encodings", __dir__)
    private_constant :DATA_DIR

    # The non-contiguous head of openai_public.py's o200k_harmony()
    # special-token table, transcribed verbatim (see PROVENANCE.md): its ten
    # named control tokens, plus the six reserved slots sitting in the gaps
    # between them (200000, 200001, 200004, 200009, 200010, 200011). The
    # reserved range only goes contiguous at 200013.
    HARMONY_HEAD_TOKENS = {
      "<|startoftext|>" => 199998,
      "<|endoftext|>" => 199999,
      "<|reserved_200000|>" => 200000,
      "<|reserved_200001|>" => 200001,
      "<|return|>" => 200002,
      "<|constrain|>" => 200003,
      "<|reserved_200004|>" => 200004,
      "<|channel|>" => 200005,
      "<|start|>" => 200006,
      "<|end|>" => 200007,
      "<|message|>" => 200008,
      "<|reserved_200009|>" => 200009,
      "<|reserved_200010|>" => 200010,
      "<|reserved_200011|>" => 200011,
      "<|call|>" => 200012,
      "<|endofprompt|>" => 200018
    }.freeze
    private_constant :HARMONY_HEAD_TOKENS

    # The contiguous tail of that reserved range: 200013..201087. With the
    # head above, 1091 entries total — 10 named, 1081 reserved (see
    # PROVENANCE.md).
    HARMONY_RESERVED_TAIL = (200013..201087).to_h { |id| ["<|reserved_#{id}|>", id] }.freeze
    private_constant :HARMONY_RESERVED_TAIL

    # Deep-frozen: entries, their `special_tokens` tables and the `rank_file` /
    # `json_file` paths. `Tokenizer#special_tokens` hands the registry's own
    # Hash back to callers, so anything less lets one caller's poke rewrite
    # what every later `from_encoding` in the process loads.
    REGISTRY = {
      "r50k_base" => {
        rank_file: File.join(DATA_DIR, "r50k_base.tiktoken"),
        pretokenizer: "gpt2",
        special_tokens: {"<|endoftext|>" => 50256}
      },
      "cl100k_base" => {
        rank_file: File.join(DATA_DIR, "cl100k_base.tiktoken"),
        pretokenizer: "gpt4",
        special_tokens: {
          "<|endoftext|>" => 100257,
          "<|fim_prefix|>" => 100258,
          "<|fim_middle|>" => 100259,
          "<|fim_suffix|>" => 100260,
          "<|endofprompt|>" => 100276
        }
      },
      "o200k_base" => {
        rank_file: File.join(DATA_DIR, "o200k_base.tiktoken"),
        pretokenizer: "o200k",
        special_tokens: {"<|endoftext|>" => 199999, "<|endofprompt|>" => 200018}
      },
      "o200k_harmony" => {
        rank_file: File.join(DATA_DIR, "o200k_base.tiktoken"),
        pretokenizer: "o200k",
        special_tokens: HARMONY_HEAD_TOKENS.merge(HARMONY_RESERVED_TAIL).freeze
      },
      # The three below are gzipped HuggingFace tokenizer.json files, vendored
      # verbatim (see PROVENANCE.md for repos, revisions and hashes). Nothing
      # else is registered: the pretokenizer and the special tokens are the
      # file's own, read at load as Tokenizer.from_json reads them.
      #
      # Qwen 3.5 and 3.6 share this one file.
      "qwen35" => {json_file: File.join(DATA_DIR, "qwen35.json.gz")},
      # The qwen35 file plus seven audio/TTS special tokens — vendored
      # separately because a JSON-backed tokenizer takes its added tokens
      # from the file.
      "qwen38" => {json_file: File.join(DATA_DIR, "qwen38.json.gz")},
      # meta-models/Muse-Glimmer-30B's file: Meta publishes no standalone
      # Muse Spark tokenizer.
      "muse_spark" => {json_file: File.join(DATA_DIR, "muse_spark.json.gz")}
    }.each_value { |encoding| encoding.each_value(&:freeze).freeze }.freeze
    private_constant :REGISTRY

    # The packaged encoding names — the single source error messages naming
    # what's available are built from (the same discipline
    # Native.pretokenizer_names / PretokenizerType::NAMES applies to
    # pretokenizer scheme names).
    NAMES = REGISTRY.keys.freeze

    # Encodings known by name but deliberately not packaged, keyed to the
    # reason a caller asking for one by name deserves to hear.
    UNPACKABLE_REASONS = {
      "p50k_base" => "its ranks are not dense (50256 is left free for <|endoftext|>), " \
        "and the rank loader rejects non-dense ranks",
      "p50k_edit" => "it loads the same p50k_base.tiktoken ranks, which are not dense " \
        "(50256 is left free for <|endoftext|>), and the rank loader rejects non-dense ranks"
    }.freeze
    private_constant :UNPACKABLE_REASONS

    class << self
      # The {rank_file:, pretokenizer:, special_tokens:} (tiktoken) or
      # {json_file:} (HuggingFace) registered for a packaged encoding name, or
      # nil. Names are Strings or Symbols, as Tokenizer.load accepts both.
      def [](name)
        REGISTRY[name.to_s]
      end

      # Why `name` can't be packaged, or nil when there's no reason on
      # record (it's either packaged, or simply not one gigatoken knows of).
      def unpackable_reason(name)
        UNPACKABLE_REASONS[name.to_s]
      end
    end
  end
end
