# frozen_string_literal: true

require_relative "../spec_helper"

RSpec.describe "gigatoken bench (integration)" do
  root = File.expand_path("../..", __dir__)

  it "runs against the GPT-2 fixture tokenizer and a fixture corpus" do
    output = IO.popen(
      %w[ruby -Ilib exe/gigatoken bench tests/fixtures/gpt2_tokenizer.json spec/fixtures/docs.txt --doc-separator <|endoftext|>],
      chdir: root, err: [:child, :out], &:read
    )
    status = $?

    expect(status).to be_success
    expect(output).to match(/\A\s*cpu: .+\n/)
    expect(output).to match(/gigatoken: +[\d.]+ s \| +[\d.]+ MB at +[\d.]+ MB\/s \| +[\d.]+ Mtok at +[\d.]+ Mtok\/s\n\z/)
  end

  it "runs against a .tiktoken fixture when given --pretokenizer" do
    output = IO.popen(
      %w[ruby -Ilib exe/gigatoken bench spec/fixtures/ranks.tiktoken spec/fixtures/docs.txt --pretokenizer gpt2],
      chdir: root, err: [:child, :out], &:read
    )
    status = $?

    expect(status).to be_success
    expect(output).to match(/gigatoken: +[\d.]+ s \| +[\d.]+ MB at +[\d.]+ MB\/s \| +[\d.]+ Mtok at +[\d.]+ Mtok\/s\n\z/)
  end

  it "fails cleanly against a .tiktoken fixture without --pretokenizer" do
    output = IO.popen(
      %w[ruby -Ilib exe/gigatoken bench spec/fixtures/ranks.tiktoken spec/fixtures/docs.txt],
      chdir: root, err: [:child, :out], &:read
    )
    status = $?

    expect(status).not_to be_success
    expect(output).to match(/error:.*pretokenizer/i)
    expect(output).not_to match(/\.rb:\d+:in /)
  end

  # A raw Errno backtrace is what a user sees for the commonest mistake of
  # all, so the shape is checked through the executable, not just the class.
  it "reports a missing FILE as one error line with no backtrace" do
    output = IO.popen(
      %w[ruby -Ilib exe/gigatoken bench cl100k_base /no/such/file.txt],
      chdir: root, err: [:child, :out], &:read
    )
    status = $?

    expect(status).not_to be_success
    expect(output.lines.grep(/\Aerror: /).size).to eq(1)
    expect(output).not_to match(/\.rb:\d+:in /)
  end

  it "reports an empty FILES list as a usage error rather than benchmarking nothing" do
    output = IO.popen(
      %w[ruby -Ilib exe/gigatoken bench cl100k_base],
      chdir: root, err: [:child, :out], &:read
    )
    status = $?

    expect(status).not_to be_success
    expect(output).to match(/\Aerror: .*FILES/)
  end
end

RSpec.describe "gigatoken validate (integration)" do
  root = File.expand_path("../..", __dir__)

  it "runs against a .tiktoken fixture when given --pretokenizer" do
    output = IO.popen(
      %w[ruby -Ilib exe/gigatoken validate spec/fixtures/ranks.tiktoken spec/fixtures/docs.txt --pretokenizer gpt2],
      chdir: root, err: [:child, :out], &:read
    )
    status = $?

    expect(status).to be_success
    expect(output).to match(/validation OK: \d+ documents match/)
  end

  it "fails cleanly against a .tiktoken fixture without --pretokenizer" do
    output = IO.popen(
      %w[ruby -Ilib exe/gigatoken validate spec/fixtures/ranks.tiktoken spec/fixtures/docs.txt],
      chdir: root, err: [:child, :out], &:read
    )
    status = $?

    expect(status).not_to be_success
    expect(output).to match(/error:.*pretokenizer/i)
    expect(output).not_to match(/\.rb:\d+:in /)
  end

  ["docs.txt.gz", "docs.txt.zst"].each do |name|
    it "validates a #{File.extname(name).delete(".")}-compressed corpus, which the Ruby-side split has to decompress too" do
      output = IO.popen(
        ["ruby", "-Ilib", "exe/gigatoken", "validate", "cl100k_base", "spec/fixtures/#{name}", "--doc-separator", "<|endoftext|>"],
        chdir: root, err: [:child, :out], &:read
      )
      status = $?

      expect(status).to be_success
      expect(output).to match(/validation OK: [1-9]\d* documents match/)
    end
  end
end
