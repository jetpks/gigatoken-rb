# frozen_string_literal: true

require_relative "lib/gigatoken/version"

Gem::Specification.new do |spec|
  spec.name = "gigatoken"
  spec.version = Gigatoken::VERSION
  spec.authors = ["Eric Jacobs"]
  spec.email = ["eric@ebj.dev"]

  spec.summary = "Tokenize your documents at GB/s"
  spec.description = "Ruby bindings to the gigatoken core crate: fast BPE/SentencePiece tokenization."
  spec.homepage = "https://github.com/jetpks/gigatoken"
  spec.license = "MIT"
  spec.metadata = {
    "bug_tracker_uri" => "https://github.com/jetpks/gigatoken/issues",
    "homepage_uri" => "https://github.com/jetpks/gigatoken",
    "source_code_uri" => "https://github.com/jetpks/gigatoken"
  }

  spec.files = Dir[
    "lib/**/*.rb",
    "ext/gigatoken/src/**/*.rs",
    "ext/gigatoken/*.{toml,rb}",
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    "src/**/*.rs",
    "README.md",
    "LICENSE"
  ]
  spec.require_paths = ["lib"]
  spec.extensions = ["ext/gigatoken/extconf.rb"]

  spec.required_ruby_version = ">= 3.3.0"
end
