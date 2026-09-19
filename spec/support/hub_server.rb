# frozen_string_literal: true

require "async"
require "async/http"
require "openssl"
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
  # composes via `Sync` without nesting), then stops the server. Given an
  # `ssl_context` it serves https, so the CONNECT tunnel has an origin to
  # speak TLS to. `Sync`, so two of these nest when an example needs an
  # origin and a CDN on separate ports.
  def run_hub_server(app, ssl_context: nil)
    port = free_port
    base_url = "#{ssl_context ? "https" : "http"}://127.0.0.1:#{port}"
    endpoint = Async::HTTP::Endpoint.parse(base_url, ssl_context: ssl_context, reuse_port: true)
    server = Async::HTTP::Server.for(endpoint, &app)

    Sync do |task|
      server_task = task.async { server.run }
      yield base_url
    ensure
      server_task&.stop
    end
  end

  # A plain-TCP CONNECT proxy on an ephemeral loopback port: `mode` :tunnel
  # splices the tunnelled bytes through to the origin, :refuse answers 407.
  # Yields its URL and the Array collecting the CONNECT request lines it saw,
  # then stops it. Threads rather than the reactor, because a tunnel is
  # opaque bytes in both directions at once.
  def run_connect_proxy(mode)
    server = TCPServer.new("127.0.0.1", 0)
    connects = []
    sockets = []
    accepting = Thread.new { loop { serve_connect(server.accept, mode, connects, sockets) } }

    yield "http://127.0.0.1:#{server.addr[1]}", connects
  ensure
    accepting&.kill
    sockets&.each { |socket| socket.close unless socket.closed? }
    server&.close
  end

  # An SSLContext serving a fresh self-signed certificate for 127.0.0.1, so
  # an example can give a CONNECT tunnel a real TLS origin to reach, and a
  # client side that trusts it. Trust goes into the process-wide default
  # store rather than through SSL_CERT_FILE: OpenSSL builds that store once,
  # when it is first required, so the variable is already read by the time
  # any example runs. Nothing else in the suite verifies a certificate, and
  # this one is generated per call and good for an hour.
  def trusted_localhost_ssl_context
    key = OpenSSL::PKey::RSA.new(2048)
    certificate = self_signed_certificate(key)
    OpenSSL::SSL::SSLContext::DEFAULT_CERT_STORE.add_cert(certificate)

    OpenSSL::SSL::SSLContext.new.tap do |context|
      context.cert = certificate
      context.key = key
    end
  end

  # A port nothing is listening on — for the refused-connection case, and
  # for binding a server to afterwards.
  def free_port
    TCPServer.open("127.0.0.1", 0) { |socket| socket.addr[1] }
  end

  private

  def self_signed_certificate(key)
    certificate = OpenSSL::X509::Certificate.new
    certificate.version = 2
    certificate.serial = 1
    certificate.subject = certificate.issuer = OpenSSL::X509::Name.parse("/CN=localhost")
    certificate.public_key = key.public_key
    certificate.not_before = Time.now - 60
    certificate.not_after = Time.now + 3600

    factory = OpenSSL::X509::ExtensionFactory.new
    factory.subject_certificate = certificate
    factory.issuer_certificate = certificate
    # A CA, because it has to verify as its own issuer.
    certificate.add_extension(factory.create_extension("basicConstraints", "CA:TRUE", true))
    certificate.add_extension(factory.create_extension("subjectAltName", "DNS:localhost,IP:127.0.0.1"))
    certificate.add_extension(factory.create_extension("keyUsage", "digitalSignature,keyEncipherment,keyCertSign", true))
    certificate.sign(key, OpenSSL::Digest.new("SHA256"))
  end

  def serve_connect(client, mode, connects, sockets)
    sockets << client
    Thread.new do
      connect_line = client.gets&.strip
      connects << connect_line
      # Drain the CONNECT request's headers, up to the blank line.
      while (header = client.gets)
        break if header == "\r\n"
      end

      if mode == :refuse
        client.write("HTTP/1.1 407 Proxy Authentication Required\r\nContent-Length: 0\r\n\r\n")
        client.close
      else
        tunnel(client, connect_line, sockets)
      end
    end
  end

  # Answer the CONNECT and shuttle bytes both ways for as long as the tunnel
  # lives.
  def tunnel(client, connect_line, sockets)
    host, port = connect_line.split[1].split(":")
    upstream = TCPSocket.new(host, port.to_i)
    sockets << upstream
    client.write("HTTP/1.1 200 Connection Established\r\n\r\n")
    [[client, upstream], [upstream, client]].each { |from, to| Thread.new { splice(from, to) } }
  end

  def splice(from, to)
    IO.copy_stream(from, to)
    to.close_write
  rescue IOError, SystemCallError
    nil
  end
end

RSpec.configure do |config|
  config.include HubServer
end
