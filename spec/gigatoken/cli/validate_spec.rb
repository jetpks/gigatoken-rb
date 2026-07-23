# frozen_string_literal: true

require_relative "../../spec_helper"
require "gigatoken/cli"
require "stringio"

RSpec.describe Gigatoken::CLI::Validate do
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

  it "reports a per-document match when encode_files agrees with encode_batch" do
    command.call(tokenizer: fixture_path, files: [docs_txt], doc_separator: "<|endoftext|>")

    expect(stdout.string).to match(/\Avalidation OK: \d+ documents match\n\z/)
  end

  it "treats a whole file with no separator as a single document" do
    command.call(tokenizer: fixture_path, files: [docs_txt])

    expect(stdout.string).to eq("validation OK: 1 documents match\n")
  end

  it "surfaces tokenizer load failures as a friendly error, exiting 1" do
    expect { command.call(tokenizer: "/no/such/tokenizer.json", files: [docs_txt]) }
      .to raise_error(SystemExit) { |e| expect(e.status).to eq(1) }
  end
end
