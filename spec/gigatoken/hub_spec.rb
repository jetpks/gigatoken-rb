# frozen_string_literal: true

require_relative "../spec_helper"
require "tmpdir"

RSpec.describe Gigatoken::Hub do
  fixture_path = File.expand_path("../../tests/fixtures/gpt2_tokenizer.json", __dir__)
  fixture = File.binread(fixture_path)
  commit = "a" * 40

  let(:app) { ->(_request) { Protocol::HTTP::Response[200, {"x-repo-commit" => commit}, [fixture]] } }

  around do |example|
    Dir.mktmpdir do |dir|
      original = ENV["HF_HOME"]
      ENV["HF_HOME"] = dir
      example.run
      ENV["HF_HOME"] = original
    end
  end

  it "downloads into the standard HF cache layout" do
    path = nil
    run_hub_server(app) do |base_url|
      path = described_class.new(endpoint: base_url).hub_file("acme/gpt2", "tokenizer.json", revision: "main")
    end

    repo_dir = described_class.repo_cache_dir("acme/gpt2")
    expect(path).to eq(repo_dir.join("snapshots", commit, "tokenizer.json"))
    expect(File.binread(path)).to eq(fixture)
    expect(File.read(repo_dir.join("refs", "main"))).to eq(commit)
  end

  it "serves a cached file with no network once downloaded" do
    hub = nil
    run_hub_server(app) do |base_url|
      hub = described_class.new(endpoint: base_url)
      hub.hub_file("acme/cached", "tokenizer.json", revision: "main")
    end

    # The server has stopped: a cache hit must not touch the network.
    path = hub.hub_file("acme/cached", "tokenizer.json", revision: "main")
    expect(File.binread(path)).to eq(fixture)
  end

  it "sends a bearer auth header when HF_TOKEN is set" do
    ENV["HF_TOKEN"] = "test-token-123"
    seen_authorization = nil
    authed_app = lambda do |request|
      seen_authorization = request.headers["authorization"]
      Protocol::HTTP::Response[200, {"x-repo-commit" => commit}, [fixture]]
    end

    run_hub_server(authed_app) do |base_url|
      described_class.new(endpoint: base_url).hub_file("acme/authed", "tokenizer.json", revision: "main")
    end

    expect(seen_authorization).to eq("Bearer test-token-123")
  ensure
    ENV.delete("HF_TOKEN")
  end

  describe ".looks_like_repo_id?" do
    it "accepts org/name and bare legacy names" do
      expect(described_class.looks_like_repo_id?("gpt2")).to be(true)
      expect(described_class.looks_like_repo_id?("openai-community/gpt2")).to be(true)
      expect(described_class.looks_like_repo_id?("Qwen/Qwen3.5-9B")).to be(true)
    end

    it "rejects paths and tokenizer file names" do
      expect(described_class.looks_like_repo_id?("data/tokenizers/gpt2.json")).to be(false)
      expect(described_class.looks_like_repo_id?("./gpt2")).to be(false)
      expect(described_class.looks_like_repo_id?("/abs/path")).to be(false)
      expect(described_class.looks_like_repo_id?("gpt2_tokenizer.json")).to be(false)
      expect(described_class.looks_like_repo_id?("subdir/tokenizer.model")).to be(false)
      expect(described_class.looks_like_repo_id?("")).to be(false)
      expect(described_class.looks_like_repo_id?("org/")).to be(false)
    end
  end
end
