# frozen_string_literal: true

require "async"
require "async/http"
require "protocol/http"
require "socket"
require "tmpdir"

# Test-only support for exercising Gigatoken::Hub against a real
# Async::HTTP::Server on loopback, so hub specs never touch the internet.
module HubServer
  # Every environment variable Gigatoken::Hub reads. Whichever one an
  # example leaves set points the cache, the token or the route at the
  # developer's real one — so they are all neutralised, and all restored.
  HUB_ENV_KEYS = %w[
    HF_HOME HF_HUB_CACHE HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_TOKEN_PATH HF_ENDPOINT
    http_proxy HTTP_PROXY https_proxy HTTPS_PROXY no_proxy NO_PROXY
  ].freeze

  # Runs the block with an empty Hub environment rooted at a fresh HF_HOME,
  # which is yielded, and restores every variable — set or unset — after.
  def with_hub_env
    saved = ENV.to_h.slice(*HUB_ENV_KEYS)
    HUB_ENV_KEYS.each { |key| ENV.delete(key) }

    Dir.mktmpdir do |home|
      ENV["HF_HOME"] = home
      yield home
    end
  ensure
    HUB_ENV_KEYS.each { |key| ENV.delete(key) }
    ENV.update(saved)
  end

  # Starts an Async::HTTP::Server for `app` on an ephemeral loopback port,
  # yields its base URL from inside a reactor (so Gigatoken::Hub#hub_file
  # composes via `Sync` without nesting), then stops the server.
  def run_hub_server(app)
    port = free_port
    endpoint = Async::HTTP::Endpoint.parse("http://127.0.0.1:#{port}", reuse_port: true)
    server = Async::HTTP::Server.for(endpoint, &app)

    Async do |task|
      server_task = task.async { server.run }
      yield "http://127.0.0.1:#{port}"
    ensure
      server_task&.stop
    end
  end

  # A port nothing is listening on — for the refused-connection case, and
  # for binding a server to afterwards.
  def free_port
    TCPServer.open("127.0.0.1", 0) { |socket| socket.addr[1] }
  end
end

RSpec.configure do |config|
  config.include HubServer
end
