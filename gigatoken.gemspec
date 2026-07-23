# frozen_string_literal: true

require_relative "lib/gigatoken/version"

Gem::Specification.new do |spec|
  spec.name = "gigatoken"
  spec.version = Gigatoken::VERSION
  spec.authors = ["Eric Jacobs"]
  spec.email = ["eric@ebj.dev"]

  spec.summary = "Tokenize your documents at GB/s"
  spec.description = "Ruby bindings to the gigatoken core crate: BPE and SentencePiece tokenization at GB/s."
  spec.homepage = "https://github.com/jetpks/gigatoken-rb"
  spec.license = "MIT"
  spec.metadata = {
    "bug_tracker_uri" => "https://github.com/jetpks/gigatoken-rb/issues",
    "homepage_uri" => "https://github.com/jetpks/gigatoken-rb",
    "source_code_uri" => "https://github.com/jetpks/gigatoken-rb"
  }

  spec.files = Dir[
    "lib/**/*.rb",
    "exe/*",
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
  spec.bindir = "exe"
  spec.executables = Dir["exe/*"].map { |f| File.basename(f) }

  spec.required_ruby_version = ">= 3.3.0"

  spec.add_dependency "async", "~> 2.43"
  spec.add_dependency "async-http", "~> 0.96"
  spec.add_dependency "dry-cli", "~> 1.0"
end
