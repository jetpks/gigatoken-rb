# frozen_string_literal: true

require "async"
require "async/http"
require "async/http/proxy"
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

    # Connect/read timeout in seconds, and the bound on the request phase —
    # huggingface_hub's HF_HUB_ETAG_TIMEOUT / HF_HUB_DOWNLOAD_TIMEOUT
    # default.
    DEFAULT_TIMEOUT = 10
    private_constant :DEFAULT_TIMEOUT

    # A "." or ".." path segment: the traversal that must never reach a URL
    # or the cache, wherever it comes from.
    DOT_SEGMENT = /\A\.\.?\z/
    private_constant :DOT_SEGMENT

    # Everything outside RFC 3986's unreserved set, which is what
    # huggingface_hub's `quote` percent-encodes.
    RESERVED = /[^A-Za-z0-9\-._~]/
    private_constant :RESERVED

    # How a request, or the body read that follows it, fails underneath
    # async-http: a connect/read timeout, a refused or reset connection, a
    # proxy refusing the CONNECT tunnel, DNS resolution, a body cut short, a
    # malformed endpoint, TLS. All of them reach the caller as
    # Gigatoken::HubError.
    TRANSPORT_ERRORS = [
      Async::TimeoutError,
      Async::HTTP::Proxy::ConnectFailure,
      IOError,
      SocketError,
      SystemCallError,
      URI::InvalidURIError,
      Protocol::HTTP::Error,
      OpenSSL::SSL::SSLError
    ].freeze
    private_constant :TRANSPORT_ERRORS

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

      # The Hub endpoint, resolved like huggingface_hub does it: HF_ENDPOINT,
      # then https://huggingface.co.
      def default_endpoint
        env("HF_ENDPOINT") || DEFAULT_ENDPOINT
      end

      # Whether `value` is usable as a URL path and a cache path component:
      # not empty, not absolute, no NUL byte, no "." or ".." segment.
      # huggingface_hub rejects the same traversals in validate_repo_id.
      def safe_component?(value)
        !value.empty? && !value.include?("\0") && !value.start_with?("/") &&
          value.split("/", -1).none? { |segment| segment.match?(DOT_SEGMENT) }
      end

      # The proxy URL the environment names for a `scheme` request to
      # `hostname`, or nil: http_proxy/https_proxy, lowercase spelling
      # first, suppressed by a matching no_proxy entry — what requests does,
      # and so what huggingface_hub inherits.
      def proxy_url(scheme, hostname)
        url = env("#{scheme}_proxy") || env("#{scheme.upcase}_PROXY")
        url unless url.nil? || no_proxy?(hostname)
      end

      private

      # no_proxy is a comma-separated list of host suffixes, or "*" for
      # everything.
      def no_proxy?(hostname)
        host = hostname.downcase
        (env("no_proxy") || env("NO_PROXY")).to_s.split(",").any? do |entry|
          entry = entry.strip.downcase.delete_prefix(".")
          next true if entry == "*"

          !entry.empty? && (host == entry || host.end_with?(".#{entry}"))
        end
      end

      def env(key)
        value = ENV[key]
        value unless value.nil? || value.empty?
      end

      def word_part?(part, first_alnum:)
        return false if part.nil? || part.empty? || part.match?(DOT_SEGMENT)

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
    #   mock); defaults to Hub.default_endpoint (HF_ENDPOINT, then
    #   huggingface.co).
    # @parameter timeout [Numeric] connect/read timeout in seconds, applied
    #   to every request and to the request phase as a whole.
    def initialize(endpoint: self.class.default_endpoint, timeout: DEFAULT_TIMEOUT)
      @endpoint = endpoint.chomp("/")
      @timeout = timeout
    end

    # Path of `filename` from Hub repo `repo_id` at `revision`, served from
    # the standard HF cache, downloading into it first when absent. The
    # three caller-supplied components are checked before any request or
    # filesystem access: one of them carrying `..` would otherwise read and
    # overwrite files outside the cache.
    def hub_file(repo_id, filename = "tokenizer.json", revision: "main")
      {"repo id" => repo_id, "filename" => filename, "revision" => revision}.each do |what, value|
        next if self.class.safe_component?(value)

        raise HubError, "#{what} #{value.inspect}: must not be empty or absolute, " \
          "or contain a NUL byte or a \".\" or \"..\" path segment"
      end

      self.class.cached_file(repo_id, filename, revision) ||
        Sync { fetch(repo_id, filename, revision) }
    end

    private

    # GET `endpoint/repo/resolve/revision/filename` and stream the body into
    # the cache snapshot named by the `x-repo-commit` response header,
    # recording the branch ref so later lookups (ours and
    # huggingface_hub's) resolve it.
    def fetch(repo_id, filename, revision)
      clients = []
      url = resolve_url(repo_id, filename, revision)
      token = self.class.hf_token
      headers = auth_headers(token)
      response = get(url, headers, clients)
      # Unlisted headers parse as a Header::Generic (an Array of values);
      # x-repo-commit is always a single value, so flatten it to a String.
      commit = response.headers["x-repo-commit"]&.to_s

      # Redirects are followed by hand: a resolve/ URL answers an LFS file
      # with a redirect to a CDN, a renamed repo with one to its new name,
      # and both headers below need a rule of their own across the hop.
      hops = 0
      while (300...400).cover?(response.status)
        location = response.headers["location"]
        response.close
        raise HubError, "#{url}: redirect with no Location header" unless location
        raise HubError, "#{url}: too many redirects" if (hops += 1) > MAX_REDIRECTS

        target = absolutize(location, url)
        # A renamed repo answers with a same-origin redirect that still needs
        # the token; the LFS CDN elsewhere authenticates by signed URL and
        # must never see it — huggingface_hub drops the header on the same
        # rule.
        headers = auth_headers(nil) unless same_origin?(target, url)
        url = target
        response = get(url, headers, clients)
        # An LFS file carries x-repo-commit on the resolve/ hop; a renamed
        # repo carries it only on the hop that finally answers 200.
        commit ||= response.headers["x-repo-commit"]&.to_s
      end
      ensure_ok!(url, response, !!token)
      ensure_commit!(url, commit)

      write_to_cache(repo_id, filename, revision, commit, response)
    rescue *TRANSPORT_ERRORS => e
      raise HubError, "#{url}: #{e.message} (#{e.class})"
    ensure
      close_all(response, clients)
    end

    # Release everything the fetch holds, in the one place every exit passes
    # through. Order matters: an unread body keeps its connection checked
    # out, and Async::HTTP::Client#close waits for its pool to drain, so the
    # response goes first and the clients innermost-first — a tunnel client's
    # connection is held by the proxy client opened under it. Bounded, and
    # deaf to the ways a broken connection fails to close: a connection some
    # failure left mid-stream would otherwise stall the drain, and with it
    # the Sync this all runs inside, for good. A leaked socket is worth less
    # than the caller's result or exception.
    def close_all(response, clients)
      Async::Task.current.with_timeout(@timeout) do
        response&.close
        clients.reverse_each(&:close)
      end
    rescue *TRANSPORT_ERRORS
      nil
    end

    # `endpoint/repo/resolve/revision/filename`, percent-encoded the way
    # huggingface_hub's `quote` does it: `revision` whole (`safe=""`), so
    # `refs/pr/1` travels as `refs%2Fpr%2F1`, while the repo id and the
    # filename keep their slashes.
    def resolve_url(repo_id, filename, revision)
      "#{@endpoint}/#{escape_path(repo_id)}/resolve/#{escape(revision)}/#{escape_path(filename)}"
    end

    def escape_path(value)
      value.split("/", -1).map { |segment| escape(segment) }.join("/")
    end

    def escape(value)
      value.gsub(RESERVED) { |char| char.bytes.map { |byte| format("%%%02X", byte) }.join }
    end

    # GET `url`, through the proxy the environment names for it when there
    # is one: an http request travels to the proxy with the absolute URI in
    # its request line, an https one through a CONNECT tunnel — how requests,
    # and so huggingface_hub, routes them. The clients stay open (the body
    # is still to be streamed); #fetch closes them.
    def get(url, headers, clients)
      endpoint = endpoint_for(url)
      proxy = self.class.proxy_url(endpoint.scheme, endpoint.hostname)
      proxy &&= endpoint_for(proxy)

      client =
        if proxy.nil?
          open_client(endpoint, clients)
        elsif endpoint.secure?
          # https tunnels through the proxy with CONNECT, then speaks TLS to
          # the origin as if it had connected to it directly.
          open_client(open_client(proxy, clients).proxied_endpoint(endpoint), clients)
        else
          open_client(proxy, clients)
        end
      # An http proxy is addressed with the absolute URI in the request line;
      # a direct or tunnelled request carries just the path.
      target = (proxy && !endpoint.secure?) ? url : endpoint.path

      request = Protocol::HTTP::Request["GET", target, headers, scheme: endpoint.scheme, authority: endpoint.authority]
      # A CONNECT tunnel's socket carries no timeout of its own, so the
      # request phase is bounded here whichever way it was routed; the body
      # then streams under the endpoint's own connect/read timeout.
      Async::Task.current.with_timeout(@timeout) { client.call(request) }
    end

    # A malformed URL comes back as URI::InvalidURIError, one that cannot be
    # routed (no scheme or host) as ArgumentError; both mean the same thing
    # to the caller, and neither is worth a backtrace.
    def endpoint_for(url)
      Async::HTTP::Endpoint.parse(url, timeout: @timeout)
    rescue ArgumentError => e
      raise HubError, "#{url}: #{e.message} (#{e.class})"
    end

    # A client for one hop, remembered in `clients` so #fetch can close it
    # once the body is written. retries: 1 — a client built for this hop has
    # no stale pooled connection for a retry to rescue, and async-http's
    # default of 3 would pay @timeout three times over.
    def open_client(endpoint, clients)
      Async::HTTP::Client.new(endpoint, retries: 1).tap { |client| clients << client }
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
      end
      File.rename(tmp, target)

      # A nested revision like refs/pr/1 is a nested ref file, so its parent
      # has to exist — huggingface_hub mkdir -p's the same path.
      if revision != commit
        ref_path = repo_dir.join("refs", revision)
        FileUtils.mkdir_p(ref_path.dirname)
        File.write(ref_path, commit)
      end

      target
    end

    # Raise on a non-success status. The unread body is left to #close_all,
    # which every exit from #fetch passes through: it keeps the HTTP/1.x
    # connection — and the Sync around it — alive until it is closed, so the
    # exception would otherwise never reach the caller.
    def ensure_ok!(url, response, had_token)
      status = response.status
      return if (200...400).cover?(status)

      case status
      when 404
        raise HubError, "#{url}: HTTP 404 — no such repo with that file, and no such local file either"
      when 401, 403
        token_note = had_token ? "the request used the discovered token" : "no token was found"
        raise HubError,
          "#{url}: HTTP #{status} — the repo may be private or gated (#{token_note}; set HF_TOKEN or run " \
          "`hf auth login`, and accept the repo's terms on huggingface.co if it is gated)"
      else
        raise HubError, "#{url}: HTTP #{status}"
      end
    end

    # The snapshot directory is named by a server-controlled header, so only
    # a real commit hash may become one: anything else would write the body
    # wherever the header points. A missing header also means the endpoint
    # isn't a Hub, whose snapshot would never be found again.
    def ensure_commit!(url, commit)
      return if self.class.commit_hash?(commit.to_s)

      raise HubError, "#{url}: response is missing a usable x-repo-commit header (#{commit.inspect}) — it does not " \
        "seem to be served by a HuggingFace Hub endpoint; if HF_ENDPOINT is set, check that it points to a " \
        "Hub-compatible endpoint, and otherwise check your firewall and proxy settings"
    end

    # A redirect Location resolved against the request URL: absolute URLs
    # pass through, host-relative (`/x/y`) and path-relative ones join the
    # base.
    def absolutize(location, base)
      return location if location.include?("://")

      base_origin = origin(base)
      return "#{base_origin}#{location}" if location.start_with?("/")

      dir_end = base.rindex("/") || base.length
      "#{base[0...[dir_end, base_origin.length].max]}/#{location}"
    end

    # Scheme and authority of a URL — everything before its path. Two URLs
    # sharing one may pass the Authorization header between them.
    def origin(url)
      scheme_end = url.index("://")
      url[0...(url.index("/", scheme_end ? scheme_end + 3 : 0) || url.length)]
    end

    def same_origin?(url, other)
      origin(url).casecmp?(origin(other))
    end
  end
end
