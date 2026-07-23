# frozen_string_literal: true

require_relative "gigatoken/version"

module Gigatoken
  # Raised for tokenizer load and encode failures surfaced from the native
  # extension — never a raw Rust panic across the Ruby boundary.
  class Error < StandardError; end

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
require_relative "gigatoken/tokenizer"
