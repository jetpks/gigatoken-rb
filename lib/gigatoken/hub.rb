# frozen_string_literal: true

require "async"
require "async/http"
require "fileutils"
require "pathname"

module Gigatoken
  # HuggingFace Hub file fetch, mirroring `huggingface_hub.hf_hub_download`:
  # same endpoint and URL layout, same token discovery (HF_TOKEN env var,
  # then the token file written by `hf auth login`), same cache directory
  # resolution — without requiring huggingface_hub, tokenizers, or
  # transformers. Files already present in the standard HF cache are served
  # with a pure-filesystem lookup (no network); on a miss the file is
  # downloaded straight into the shared cache, so later loads (ours and
  # huggingface_hub's) are served from it.
  #
  # Network I/O runs on async-http, on the reactor: #hub_file wraps its
  # fetch in `Sync`, so it composes whether the caller is already inside a
  # reactor or is plain sync code.
  class Hub
    # Filename suffixes of local tokenizer files (tokenizer.json contents
    # and raw sentencepiece models). A name ending in one of these is never
    # treated as a Hub repo id, so a mistyped local path fails fast instead
    # of hitting the network. Keep in sync with `src/load_tokenizer/hub.rs`'s
    # TOKENIZER_FILE_SUFFIXES.
    TOKENIZER_FILE_SUFFIXES = [".json", ".model"].freeze

    DEFAULT_ENDPOINT = "https://huggingface.co"
    private_constant :DEFAULT_ENDPOINT

    MAX_REDIRECTS = 10
    private_constant :MAX_REDIRECTS

    class << self
      # Whether `name` is shaped like a HuggingFace Hub repo id: `org/name`,
      # or a bare legacy repo name like `gpt2`. At most one slash, and not
      # something that is obviously a filesystem path to a local tokenizer
      # file.
      def looks_like_repo_id?(name)
        parts = name.split("/", -1)
        return false if parts.empty? || parts.size > 2

        org, rest = parts
        return false unless word_part?(org, first_alnum: true)
        return false if rest && !word_part?(rest, first_alnum: false)

        TOKENIZER_FILE_SUFFIXES.none? { |suffix| name.end_with?(suffix) }
      end

      # $HF_HOME, defaulting to $XDG_CACHE_HOME/huggingface then
      # ~/.cache/huggingface — the root for both the hub cache and the token
      # file.
      def hf_home
        Pathname.new(env("HF_HOME") || File.join(env("XDG_CACHE_HOME") || File.join(Dir.home, ".cache"), "huggingface"))
      end

      # The standard HuggingFace hub cache directory, resolved like
      # huggingface_hub does it: HF_HUB_CACHE, then $HF_HOME/hub.
      def hf_hub_cache_dir
        Pathname.new(env("HF_HUB_CACHE") || hf_home.join("hub").to_s)
      end

      # The HuggingFace access token, discovered like huggingface_hub does
      # it: the HF_TOKEN (or legacy HUGGING_FACE_HUB_TOKEN) environment
      # variable, then the token file (HF_TOKEN_PATH, default $HF_HOME/token).
      def hf_token
        token = env("HF_TOKEN") || env("HUGGING_FACE_HUB_TOKEN")
        return token.strip if token

        token_path = env("HF_TOKEN_PATH") || hf_home.join("token").to_s
        return nil unless File.file?(token_path)

        token = File.read(token_path).strip
        token unless token.empty?
      end

      # Path of `filename` in the local HF cache, or nil when not cached. A
      # pure-filesystem lookup — no request is made. `revision` may be a
      # commit hash (used directly as the snapshot name) or a branch/tag
      # name (followed through the cached ref).
      def cached_file(repo_id, filename, revision)
        repo_dir = repo_cache_dir(repo_id)
        commit = commit_hash?(revision) ? revision : cached_ref(repo_dir, revision)
        return nil unless commit

        path = repo_dir.join("snapshots", commit, filename)
        path if path.file?
      end

      # The cache directory of a repo (`models--org--name`).
      def repo_cache_dir(repo_id)
        hf_hub_cache_dir.join("models--#{repo_id.gsub("/", "--")}")
      end

      # A full git commit hash: cache snapshot directories are named by
      # these.
      def commit_hash?(revision)
        revision.match?(/\A[0-9a-f]{40}\z/)
      end

      private

      def env(key)
        value = ENV[key]
        value unless value.nil? || value.empty?
      end

      def word_part?(part, first_alnum:)
        return false if part.nil? || part.empty?

        first_ok = first_alnum ? part[0].match?(/[A-Za-z0-9]/) : word_char?(part[0])
        first_ok && part[1..].chars.all? { |c| word_char?(c) }
      end

      def word_char?(char)
        char.match?(/[A-Za-z0-9_.-]/)
      end

      def cached_ref(repo_dir, revision)
        ref_path = repo_dir.join("refs", revision)
        File.read(ref_path).strip if ref_path.file?
      end
    end

    # @parameter endpoint [String] the Hub endpoint to fetch from — override
    #   for pointing at a local server in tests (dependency injection, not a
    #   mock).
    def initialize(endpoint: DEFAULT_ENDPOINT)
      @endpoint = endpoint.chomp("/")
      @internet = Async::HTTP::Internet.new
    end

    # Path of `filename` from Hub repo `repo_id` at `revision`, served from
    # the standard HF cache, downloading into it first when absent.
    def hub_file(repo_id, filename = "tokenizer.json", revision: "main")
      self.class.cached_file(repo_id, filename, revision) ||
        Sync { fetch(repo_id, filename, revision) }
    end

    private

    # GET `endpoint/repo/resolve/revision/filename` and stream the body into
    # the cache snapshot named by the `x-repo-commit` response header,
    # recording the branch ref so later lookups (ours and
    # huggingface_hub's) resolve it.
    def fetch(repo_id, filename, revision)
      url = "#{@endpoint}/#{repo_id}/resolve/#{revision}/#{filename}"
      token = self.class.hf_token
      response = @internet.get(url, auth_headers(token))
      # Unlisted headers parse as a Header::Generic (an Array of values);
      # x-repo-commit is always a single value, so flatten it to a String.
      commit = response.headers["x-repo-commit"]&.to_s

      # Redirects are followed by hand: resolve/ URLs answer with the
      # x-repo-commit header and a redirect to a CDN for LFS files, and the
      # Authorization header must not travel to the other host.
      hops = 0
      while (300...400).cover?(response.status)
        ensure_ok!(url, response.status, !!token)
        hops += 1
        raise Error, "#{url}: too many redirects" if hops > MAX_REDIRECTS

        location = response.headers["location"] ||
          raise(Error, "#{url}: redirect with no Location header")
        response.close
        url = absolutize(location, url)
        response = @internet.get(url, {"user-agent" => "gigatoken"})
      end
      ensure_ok!(url, response.status, !!token)

      write_to_cache(repo_id, filename, revision, commit || revision, response)
    end

    def auth_headers(token)
      headers = {"user-agent" => "gigatoken"}
      headers["authorization"] = "Bearer #{token}" if token
      headers
    end

    # Stream the response body to a sibling temp file, then rename into
    # place: concurrent downloaders race benignly and readers never observe
    # a partial file.
    def write_to_cache(repo_id, filename, revision, commit, response)
      repo_dir = self.class.repo_cache_dir(repo_id)
      target = repo_dir.join("snapshots", commit, filename)
      FileUtils.mkdir_p(target.dirname)

      tmp = target.dirname.join(".#{target.basename}.#{Process.pid}.tmp")
      begin
        response.save(tmp.to_s)
      rescue
        FileUtils.rm_f(tmp)
        raise
      ensure
        response.close
      end
      File.rename(tmp, target)

      if !self.class.commit_hash?(revision) && revision != commit
        refs_dir = repo_dir.join("refs")
        FileUtils.mkdir_p(refs_dir)
        File.write(refs_dir.join(revision), commit)
      end

      target
    end

    def ensure_ok!(url, status, had_token)
      return if (200...400).cover?(status)

      case status
      when 404
        raise Error, "#{url}: HTTP 404 — no such repo with that file, and no such local file either"
      when 401, 403
        token_note = had_token ? "the request used the discovered token" : "no token was found"
        raise Error,
          "#{url}: HTTP #{status} — the repo may be private or gated (#{token_note}; set HF_TOKEN or run " \
          "`hf auth login`, and accept the repo's terms on huggingface.co if it is gated)"
      else
        raise Error, "#{url}: HTTP #{status}"
      end
    end

    # A redirect Location resolved against the request URL: absolute URLs
    # pass through, host-relative (`/x/y`) and path-relative ones join the
    # base.
    def absolutize(location, base)
      return location if location.include?("://")

      origin_end = base.index("://") ? base.index("://") + 3 : 0
      origin_end = base.index("/", origin_end) || base.length

      if location.start_with?("/")
        "#{base[0...origin_end]}#{location}"
      else
        dir_end = base.rindex("/") || base.length
        "#{base[0...[dir_end, origin_end].max]}/#{location}"
      end
    end
  end
end
