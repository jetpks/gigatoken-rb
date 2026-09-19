# frozen_string_literal: true

require_relative "../../spec_helper"
require "gigatoken/cli"
require "stringio"
require "tmpdir"
require "zlib"

RSpec.describe Gigatoken::CLI::Bench do
  fixtures = File.expand_path("../../fixtures", __dir__)
  fixture_path = File.expand_path("../../../tests/fixtures/gpt2_tokenizer.json", __dir__)
  docs_txt = File.join(fixtures, "docs.txt")

  let(:stdout) { StringIO.new }
  let(:stderr) { StringIO.new }
  let(:command) do
    described_class.new.tap do |cmd|
      cmd.instance_variable_set(:@out, stdout)
      cmd.instance_variable_set(:@err, stderr)
    end
  end

  # The MB figure from the throughput line just printed, clearing the capture
  # for the next call (StringIO#truncate leaves the write position behind).
  def reported_mb
    mb = stdout.string[/([\d.]+) MB at/, 1]
    stdout.truncate(0)
    stdout.rewind
    mb
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

  # MB/s over the compressed size would understate throughput by the whole
  # compression ratio, and the batch path would be tokenizing gzip bytes.
  it "reports MB over the decompressed bytes of a .gz input, on every path" do
    Dir.mktmpdir do |dir|
      text = "the quick brown fox jumps over the lazy dog. " * 20_000
      plain = File.join(dir, "docs.txt")
      File.write(plain, text)
      Zlib::GzipWriter.open("#{plain}.gz") { |gz| gz.write(text) }

      [{}, {packed: true}, {parallel: false}].each do |options|
        command.call(tokenizer: fixture_path, files: ["#{plain}.gz"], **options)
        gz_mb = reported_mb
        command.call(tokenizer: fixture_path, files: [plain], **options)

        expect(gz_mb).to eq(reported_mb)
        expect(gz_mb.to_f).to be > 0.5
      end
    end
  end

  it "refuses .zst, naming the missing decoder and the native path that handles it" do
    Dir.mktmpdir do |dir|
      zst = File.join(dir, "docs.txt.zst")
      File.binwrite(zst, "not really zstd")

      expect { command.call(tokenizer: fixture_path, files: [zst]) }
        .to raise_error(SystemExit) { |e| expect(e.status).to eq(1) }
      expect(stderr.string).to match(/\Aerror: .*zstd.*encode_files/)
    end
  end

  it "prints one error line and exits 1 for a missing FILE" do
    expect { command.call(tokenizer: fixture_path, files: ["/no/such/file.txt"]) }
      .to raise_error(SystemExit) { |e| expect(e.status).to eq(1) }
    expect(stderr.string).to match(/\Aerror: .+\n\z/)
  end

  it "prints one error line and exits 1 for a directory given as a FILE" do
    Dir.mktmpdir do |dir|
      expect { command.call(tokenizer: fixture_path, files: [dir]) }
        .to raise_error(SystemExit) { |e| expect(e.status).to eq(1) }
      expect(stderr.string).to match(/\Aerror: .+\n\z/)
    end
  end

  it "prints one error line and exits 1 for a .gz that is not gzip" do
    Dir.mktmpdir do |dir|
      corrupt = File.join(dir, "docs.txt.gz")
      File.binwrite(corrupt, "not really gzip")

      expect { command.call(tokenizer: fixture_path, files: [corrupt]) }
        .to raise_error(SystemExit) { |e| expect(e.status).to eq(1) }
      expect(stderr.string).to match(/\Aerror: .+\n\z/)
    end
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

  describe "--limit-bytes units" do
    it "reads KiB/MiB/GiB/TiB as binary and KB/MB/GB/TB as decimal" do
      expect(Gigatoken::CLI::Support.parse_size("1KiB")).to eq(1024)
      expect(Gigatoken::CLI::Support.parse_size("1KB")).to eq(1000)
      expect(Gigatoken::CLI::Support.parse_size("2MiB")).to eq(2 * 1024**2)
      expect(Gigatoken::CLI::Support.parse_size("2MB")).to eq(2_000_000)
      expect(Gigatoken::CLI::Support.parse_size("1500")).to eq(1500)
    end

    it "treats none and unlimited as no cap" do
      expect(Gigatoken::CLI::Support.parse_size("none")).to be_nil
      expect(Gigatoken::CLI::Support.parse_size("unlimited")).to be_nil
    end

    it "raises Gigatoken::Error naming the unparseable size" do
      expect { Gigatoken::CLI::Support.parse_size("1e3") }.to raise_error(Gigatoken::Error, /1e3/)
    end
  end
end
