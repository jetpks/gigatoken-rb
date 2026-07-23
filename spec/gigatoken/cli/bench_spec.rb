# frozen_string_literal: true

require_relative "../../spec_helper"
require "gigatoken/cli"
require "stringio"

RSpec.describe Gigatoken::CLI::Bench do
  fixtures = File.expand_path("../../fixtures", __dir__)
  fixture_path = File.expand_path("../../../tests/fixtures/gpt2_tokenizer.json", __dir__)
  docs_txt = File.join(fixtures, "docs.txt")

  let(:stdout) { StringIO.new }
  let(:command) do
    described_class.new.tap do |cmd|
      cmd.instance_variable_set(:@out, stdout)
      cmd.instance_variable_set(:@err, StringIO.new)
    end
  end

  it "prints the cpu line and the gigatoken throughput line in the Python CLI's shape" do
    command.call(tokenizer: fixture_path, files: [docs_txt], doc_separator: "<|endoftext|>")

    expect(stdout.string).to match(/\A\s*cpu: .+\n/)
    expect(stdout.string).to match(/gigatoken: +[\d.]+ s \| +[\d.]+ MB at +[\d.]+ MB\/s \| +[\d.]+ Mtok at +[\d.]+ Mtok\/s\n\z/)
  end

  it "encodes the same token counts whether run serially or on the worker pool" do
    command.call(tokenizer: fixture_path, files: [docs_txt], doc_separator: "<|endoftext|>", parallel: false)
    serial_tokens = stdout.string[/([\d.]+) Mtok at/, 1]

    stdout.truncate(0)
    command.call(tokenizer: fixture_path, files: [docs_txt], doc_separator: "<|endoftext|>", parallel: true)
    parallel_tokens = stdout.string[/([\d.]+) Mtok at/, 1]

    expect(serial_tokens).to eq(parallel_tokens)
  end

  it "caps the benchmarked bytes with limit_bytes" do
    command.call(tokenizer: fixture_path, files: [docs_txt], limit_bytes: "1B")

    expect(stdout.string).to match(/ +0\.00 MB at/)
  end

  it "defaults limit_bytes to 'none' (uncapped)" do
    limit_bytes_option = described_class.options.find { |option| option.name == :limit_bytes }

    expect(limit_bytes_option.default).to eq("none")
  end

  it "defaults packed to false" do
    packed_option = described_class.options.find { |option| option.name == :packed }

    expect(packed_option.default).to eq(false)
  end

  it "prints the gigatoken throughput line with --packed" do
    command.call(tokenizer: fixture_path, files: [docs_txt], doc_separator: "<|endoftext|>", packed: true)

    expect(stdout.string).to match(/gigatoken: +[\d.]+ s \| +[\d.]+ MB at +[\d.]+ MB\/s \| +[\d.]+ Mtok at +[\d.]+ Mtok\/s\n\z/)
  end

  it "reports the same token count packed as unpacked" do
    command.call(tokenizer: fixture_path, files: [docs_txt], doc_separator: "<|endoftext|>", parallel: false)
    unpacked_tokens = stdout.string[/([\d.]+) Mtok at/, 1]

    stdout.truncate(0)
    command.call(tokenizer: fixture_path, files: [docs_txt], doc_separator: "<|endoftext|>", packed: true)
    packed_tokens = stdout.string[/([\d.]+) Mtok at/, 1]

    expect(packed_tokens).to eq(unpacked_tokens)
  end

  it "encodes the same token counts whether packed runs serially or on the worker pool" do
    command.call(tokenizer: fixture_path, files: [docs_txt], doc_separator: "<|endoftext|>", packed: true, parallel: false)
    serial_tokens = stdout.string[/([\d.]+) Mtok at/, 1]

    stdout.truncate(0)
    command.call(tokenizer: fixture_path, files: [docs_txt], doc_separator: "<|endoftext|>", packed: true, parallel: true)
    parallel_tokens = stdout.string[/([\d.]+) Mtok at/, 1]

    expect(serial_tokens).to eq(parallel_tokens)
  end

  it "surfaces tokenizer load failures as a friendly error, exiting 1" do
    expect { command.call(tokenizer: "/no/such/tokenizer.json", files: [docs_txt]) }
      .to raise_error(SystemExit) { |e| expect(e.status).to eq(1) }
  end
end
