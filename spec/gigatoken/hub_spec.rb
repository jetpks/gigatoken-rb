# frozen_string_literal: true

require_relative "../spec_helper"

RSpec.describe Gigatoken::Hub do
  fixture_path = File.expand_path("../../tests/fixtures/gpt2_tokenizer.json", __dir__)
  fixture = File.binread(fixture_path)
  commit = "a" * 40

  let(:app) { ->(_request) { Protocol::HTTP::Response[200, {"x-repo-commit" => commit}, [fixture]] } }

  around do |example|
    with_hub_env do |home|
      @home = home
      example.run
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
  end

  it "downloads a nested revision once, percent-encoded, then hits the cache" do
    paths = []
    counting_app = lambda do |request|
      paths << request.path
      Protocol::HTTP::Response[200, {"x-repo-commit" => commit}, [fixture]]
    end

    path = nil
    run_hub_server(counting_app) do |base_url|
      hub = described_class.new(endpoint: base_url)
      path = hub.hub_file("acme/pr", revision: "refs/pr/1")
      hub.hub_file("acme/pr", revision: "refs/pr/1")
    end

    expect(paths).to eq(["/acme/pr/resolve/refs%2Fpr%2F1/tokenizer.json"])
    expect(File.binread(path)).to eq(fixture)
    expect(File.read(described_class.repo_cache_dir("acme/pr").join("refs", "refs", "pr", "1"))).to eq(commit)
  end

  describe "server responses" do
    it "raises on an error status carrying a body, rather than hanging on the unread body" do
      run_hub_server(->(_request) { Protocol::HTTP::Response[404, {}, ["no such file"]] }) do |base_url|
        expect { described_class.new(endpoint: base_url).hub_file("acme/missing") }
          .to raise_error(Gigatoken::Error, /HTTP 404/)
      end
    end

    it "raises on a 500 with a body" do
      run_hub_server(->(_request) { Protocol::HTTP::Response[500, {}, ["boom"]] }) do |base_url|
        expect { described_class.new(endpoint: base_url).hub_file("acme/broken") }
          .to raise_error(Gigatoken::Error, /HTTP 500/)
      end
    end

    it "raises when the response has no x-repo-commit header, rather than caching an unfindable snapshot" do
      run_hub_server(->(_request) { Protocol::HTTP::Response[200, {}, [fixture]] }) do |base_url|
        expect { described_class.new(endpoint: base_url).hub_file("acme/nocommit") }
          .to raise_error(Gigatoken::Error, /x-repo-commit/)
      end
    end

    it "refuses an x-repo-commit that is not a commit hash, writing nothing outside HF_HOME" do
      traversing = ->(_request) { Protocol::HTTP::Response[200, {"x-repo-commit" => "../../../../pwn"}, [fixture]] }

      run_hub_server(traversing) do |base_url|
        expect { described_class.new(endpoint: base_url).hub_file("acme/hostile") }
          .to raise_error(Gigatoken::Error, /x-repo-commit/)
      end

      expect(File.exist?(File.join(@home, "..", "pwn"))).to be(false)
    end
  end

  describe "caller input" do
    it "rejects traversing revisions and filenames before making a request" do
      requests = 0
      counting_app = lambda do |_request|
        requests += 1
        Protocol::HTTP::Response[200, {"x-repo-commit" => commit}, [fixture]]
      end

      run_hub_server(counting_app) do |base_url|
        hub = described_class.new(endpoint: base_url)
        expect { hub.hub_file("acme/x", revision: "../../x") }.to raise_error(Gigatoken::Error, /revision/)
        expect { hub.hub_file("acme/x", "../x") }.to raise_error(Gigatoken::Error, /filename/)
        expect { hub.hub_file("../acme/x") }.to raise_error(Gigatoken::Error, /repo id/)
        expect { hub.hub_file("acme/x", "/etc/passwd") }.to raise_error(Gigatoken::Error, /filename/)
        expect { hub.hub_file("acme/x", "tokenizer\0.json") }.to raise_error(Gigatoken::Error, /filename/)
      end

      expect(requests).to eq(0)
    end
  end

  describe "transport failures" do
    it "raises Gigatoken::Error when the connection is refused" do
      hub = described_class.new(endpoint: "http://127.0.0.1:#{free_port}")
      expect { hub.hub_file("acme/x") }.to raise_error(Gigatoken::Error, /Connection refused/)
    end

    it "raises Gigatoken::Error when the server accepts but never answers" do
      server = TCPServer.new("127.0.0.1", 0)
      accepted = []
      accepting = Thread.new { loop { accepted << server.accept } }

      hub = described_class.new(endpoint: "http://127.0.0.1:#{server.addr[1]}", timeout: 0.2)
      expect { hub.hub_file("acme/slow") }.to raise_error(Gigatoken::Error, /timeout|expired/i)
    ensure
      accepting&.kill
      accepted&.each(&:close)
      server&.close
    end
  end

  describe "proxy environment" do
    it "sends an http request to http_proxy with the absolute URI in its request line" do
      seen = nil
      proxy_app = lambda do |request|
        # An absolute-URI request line is split into scheme/authority/path by
        # the server; a relative one would leave the scheme nil.
        seen = [request.scheme, request.authority, request.path]
        Protocol::HTTP::Response[200, {"x-repo-commit" => commit}, [fixture]]
      end

      path = nil
      run_hub_server(proxy_app) do |proxy_url|
        ENV["http_proxy"] = proxy_url
        # The endpoint host does not resolve: only the proxy can answer this.
        path = described_class.new(endpoint: "http://example.invalid").hub_file("acme/proxied")
      end

      expect(seen).to eq(["http", "example.invalid", "/acme/proxied/resolve/main/tokenizer.json"])
      expect(File.binread(path)).to eq(fixture)
    end

    it "goes direct when no_proxy names the host" do
      path = nil
      run_hub_server(app) do |base_url|
        ENV["http_proxy"] = "http://127.0.0.1:#{free_port}"
        ENV["no_proxy"] = "127.0.0.1"
        path = described_class.new(endpoint: base_url).hub_file("acme/bypassed")
      end

      expect(File.binread(path)).to eq(fixture)
    end
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

    it "rejects dot segments in either half" do
      expect(described_class.looks_like_repo_id?("org/..")).to be(false)
      expect(described_class.looks_like_repo_id?("org/.")).to be(false)
      expect(described_class.looks_like_repo_id?("../x")).to be(false)
      expect(described_class.looks_like_repo_id?("..")).to be(false)
    end
  end
end
