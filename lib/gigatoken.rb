# frozen_string_literal: true

require_relative "gigatoken/version"

module Gigatoken
  # Raised for tokenizer load and encode failures surfaced from the native
  # extension — never a raw Rust panic across the Ruby boundary. The base of
  # the three below: rescue this to catch everything gigatoken raises.
  # Anything narrower than the three has no class of its own — CLI usage
  # errors and the odd leftover are this one directly.
  class Error < StandardError; end

  # Everything Gigatoken::Hub raises: HTTP status, transport, timeout, repo-id
  # / revision / filename / x-repo-commit validation.
  class HubError < Error; end

  # A document the tokenizer cannot take: an untranscodable or invalid-byte
  # String, invalid UTF-8 on the SentencePiece path, an id outside the
  # vocabulary in #decode.
  class InputError < Error; end

  # A tokenizer that cannot be loaded: bad or hostile JSON, a missing file or
  # directory, an unknown or unpackable encoding name, a malformed .tiktoken.
  class ModelError < Error; end

  class << self
    # The process-global encode-cache budget in bytes per worker (a parallel
    # batch encode may use up to workers x budget), applied to tokenizers of
    # either backend constructed *afterward* — changing it never affects an
    # already-built Tokenizer. nil means unbounded. Default: 512 MiB.
    def max_cache_bytes
      Native.get_max_cache_bytes
    end

    def max_cache_bytes=(bytes)
      Native.set_max_cache_bytes(bytes)
    end
  end

  NATIVE_EXTENSIONS = %w[.bundle .so .rb].freeze

  # Precompiled native gems ship per-ABI subdirs (`gigatoken/4.0/...`),
  # the source-gem `rake compile` build lands flat (`gigatoken/...`).
  # Pick whichever exists for the current Ruby ABI, with the per-ABI path
  # winning when both are present.
  def self.locate_native(base, ruby_version: RUBY_VERSION)
    abi = ruby_version[/\d+\.\d+/]
    candidates = [File.join(base, abi, "gigatoken_rb"), File.join(base, "gigatoken_rb")]
    candidates.find { |stem| NATIVE_EXTENSIONS.any? { |ext| File.exist?(stem + ext) } }
  end
end

native = Gigatoken.locate_native(File.expand_path("gigatoken", __dir__))
raise LoadError, "could not locate gigatoken native extension" unless native
require native

require_relative "gigatoken/hub"
require_relative "gigatoken/packed_result"
require_relative "gigatoken/encodings"
require_relative "gigatoken/tokenizer"
