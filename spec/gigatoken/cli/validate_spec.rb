# frozen_string_literal: true

require_relative "../../spec_helper"
require "gigatoken/cli"
require "stringio"
require "tmpdir"

RSpec.describe Gigatoken::CLI::Validate do
  fixtures = File.expand_path("../../fixtures", __dir__)
  fixture_path = File.expand_path("../../../tests/fixtures/gpt2_tokenizer.json", __dir__)
  docs_txt = File.join(fixtures, "docs.txt")
  compressed = {gz: File.join(fixtures, "docs.txt.gz"), zst: File.join(fixtures, "docs.txt.zst")}

  let(:stdout) { StringIO.new }
  let(:stderr) { StringIO.new }
  let(:command) do
    described_class.new.tap do |cmd|
      cmd.instance_variable_set(:@out, stdout)
      cmd.instance_variable_set(:@err, stderr)
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

  # The whole point of validate is that both sides see the same bytes: the
  # native source decompresses, so the Ruby-side split has to as well or a
  # correct tokenizer "fails" on every compressed corpus.
  compressed.each do |kind, path|
    it "agrees on a .#{kind} corpus, matching the uncompressed run document for document" do
      command.call(tokenizer: fixture_path, files: [path], doc_separator: "<|endoftext|>")
      from_compressed = stdout.string.dup

      stdout.truncate(0)
      stdout.rewind
      command.call(tokenizer: fixture_path, files: [docs_txt], doc_separator: "<|endoftext|>")

      expect(from_compressed).to match(/\Avalidation OK: [1-9]\d* documents match\n\z/)
      expect(from_compressed).to eq(stdout.string)
    end
  end

  it "prints one error line and exits 1 for a missing FILE" do
    expect { command.call(tokenizer: fixture_path, files: ["/no/such/file.txt"]) }
      .to raise_error(SystemExit) { |e| expect(e.status).to eq(1) }
    expect(stderr.string).to match(/\Aerror: .+\n\z/)
  end

  it "rejects an empty FILES list as a usage error" do
    expect { command.call(tokenizer: fixture_path, files: []) }
      .to raise_error(SystemExit) { |e| expect(e.status).to eq(1) }
    expect(stderr.string).to match(/\Aerror: .*FILES/)
  end

  it "rejects an empty --doc-separator as a usage error" do
    expect { command.call(tokenizer: fixture_path, files: [docs_txt], doc_separator: "") }
      .to raise_error(SystemExit) { |e| expect(e.status).to eq(1) }
    expect(stderr.string).to match(/\Aerror: .*doc-separator/)
  end
end
