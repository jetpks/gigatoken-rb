# frozen_string_literal: true

require "async"
require "async/http"
require "protocol/http"
require "socket"

# Test-only support for exercising Gigatoken::Hub against a real
# Async::HTTP::Server on loopback, so hub specs never touch the internet.
module HubServer
  # Starts an Async::HTTP::Server for `app` on an ephemeral loopback port,
  # yields its base URL from inside a reactor (so Gigatoken::Hub#hub_file
  # composes via `Sync` without nesting), then stops the server.
  def run_hub_server(app)
    port = TCPServer.open("127.0.0.1", 0) { |socket| socket.addr[1] }
    endpoint = Async::HTTP::Endpoint.parse("http://127.0.0.1:#{port}", reuse_port: true)
    server = Async::HTTP::Server.for(endpoint, &app)

    Async do |task|
      server_task = task.async { server.run }
      yield "http://127.0.0.1:#{port}"
    ensure
      server_task&.stop
    end
  end
end

RSpec.configure do |config|
  config.include HubServer
end
