# frozen_string_literal: true

require "etc"
require "zlib"

module Gigatoken
  module CLI
    # Helpers shared by the bench and validate commands: argument checking,
    # tokenizer loading, byte-size parsing, document splitting, and CPU
    # identification.
    module Support
      SIZE_UNITS = {"" => 1, "k" => 10**3, "m" => 10**6, "g" => 10**9, "t" => 10**12}.freeze
      private_constant :SIZE_UNITS

      # "KiB" and friends are binary, as everywhere else that spells the i.
      BINARY_SIZE_UNITS = {"" => 1, "k" => 2**10, "m" => 2**20, "g" => 2**30, "t" => 2**40}.freeze
      private_constant :BINARY_SIZE_UNITS

      SIZE_PATTERN = /\A\s*(\d+(?:\.\d+)?)\s*([kmgt]?)(i?)b?\s*\z/i
      private_constant :SIZE_PATTERN

      # Compression is detected from the extension, the same way the native
      # file sources do it (src/input/file_source.rs `detect_compression`) —
      # the Ruby-side split has to agree with them or the two paths see
      # different bytes.
      ZSTD_SUFFIXES = [".zst", ".zstd"].freeze
      private_constant :ZSTD_SUFFIXES

      READ_CHUNK_BYTES = 1 << 20
      private_constant :READ_CHUNK_BYTES

      class << self
        # The argument shapes dry-cli itself lets through: FILES declared
        # `required: true` still arrives empty, and an empty separator would
        # split every byte into its own document. A .zst input is refused
        # here rather than part-way through, so neither command does work it
        # cannot finish.
        def check_usage!(files, separator)
          raise Gigatoken::Error, "FILES is required: name at least one file to encode" if files.empty?
          raise Gigatoken::Error, "--doc-separator cannot be empty" if separator == ""

          files.each { |file| refuse_zstd!(file.to_s) }
        end

        # Load TOKENIZER: a tokenizer.json path/directory, a packaged
        # tiktoken encoding name, a HuggingFace repo id, or a .tiktoken file
        # — see Gigatoken::Tokenizer.load. `pretokenizer:` is forwarded
        # as-is; it's required for a bare .tiktoken path (which carries no
        # scheme of its own) and ignored for the other shapes.
        def load_tokenizer(spec, pretokenizer: nil)
          Gigatoken::Tokenizer.load(spec, pretokenizer: pretokenizer)
        end

        # Parse a byte size like "100MB", "2.5GB", "64KiB" or "1000000" —
        # decimal units unless the prefix spells the i, which makes it
        # binary; "none"/"unlimited" means no limit.
        def parse_size(text)
          return nil if ["none", "unlimited"].include?(text.strip.downcase)

          match = SIZE_PATTERN.match(text)
          raise Gigatoken::Error, "cannot parse size #{text.inspect}; expected something like 100MB (or 'none')" unless match

          units = match[3].empty? ? SIZE_UNITS : BINARY_SIZE_UNITS
          (match[1].to_f * units.fetch(match[2].downcase)).to_i
        end

        # A Native::TextFileSource for FILES, split on `separator` when
        # given.
        def text_file_source(files, separator)
          Gigatoken::Native::TextFileSource.new(files.map(&:to_s), separator: separator)
        end

        # Whole files as raw bytes, one document per file, or (with a
        # separator) the separator-split pieces of each file in order, empty
        # documents skipped. Compressed files are decompressed first, as the
        # native file sources do, so both paths split the same bytes.
        def split_docs(files, separator)
          raws = files.map { |file| read_decompressed(file.to_s) }
          return raws if separator.nil?

          sep = separator.b
          raws.flat_map { |raw| raw.split(sep).reject(&:empty?) }
        end

        # The bytes the tokenizer actually sees for FILES — a compressed
        # file's decompressed size, not its size on disk — so throughput is
        # reported over the input that was tokenized.
        def input_bytesize(files)
          files.sum { |file| decompressed_size(file.to_s) }
        end

        # The prefix of `docs` totalling at most `limit_bytes`, byte-
        # truncating the final document to fill the budget. Unlike a
        # text-comparison tool, gigatoken encodes raw bytes and does not
        # require the cut to land on a UTF-8 character boundary.
        def subset_docs(docs, limit_bytes)
          return docs if limit_bytes.nil?

          subset = []
          used = 0
          docs.each do |doc|
            room = limit_bytes - used
            if doc.bytesize <= room
              subset << doc
              used += doc.bytesize
            else
              subset << doc.byteslice(0, room) if room > 0
              break
            end
          end
          subset
        end

        # The benchmark machine's CPU as "name, N cores", plus ", M sockets"
        # when there is more than one socket.
        def cpu_info
          name, cores, sockets =
            case RbConfig::CONFIG["host_os"]
            when /darwin/ then darwin_cpu_info
            when /linux/ then linux_cpu_info
            end
          name ||= RbConfig::CONFIG["host_cpu"] || "unknown CPU"
          cores ||= Etc.nprocessors
          parts = [name, "#{cores} core#{"s" unless cores == 1}"]
          parts << "#{sockets} sockets" if sockets && sockets > 1
          parts.join(", ")
        end

        private

        # gigatoken-rb has no Ruby-side zstd decoder — `zstd-ruby` is not a
        # dependency — so the commands that need the documents (or their
        # size) in Ruby cannot handle .zst. The library's own
        # `Tokenizer#encode_files` decompresses it natively and is
        # unaffected; decompress the file first to bench or validate it.
        def read_decompressed(path)
          refuse_zstd!(path)
          return File.binread(path) unless gzip?(path)

          gunzip(path) { |gz| gz.read.b }
        end

        # The decompressed size without materializing the bytes: --packed
        # exists so a large corpus never becomes Ruby Strings, and counting
        # it must not undo that.
        def decompressed_size(path)
          refuse_zstd!(path)
          return File.size(path) unless gzip?(path)

          gunzip(path) do |gz|
            size = 0
            size += gz.read(READ_CHUNK_BYTES).bytesize until gz.eof?
            size
          end
        end

        def gunzip(path, &block)
          Zlib::GzipReader.open(path, &block)
        rescue Zlib::Error => e
          raise Gigatoken::Error, "#{path}: #{e.message}"
        end

        def refuse_zstd!(path)
          return unless ZSTD_SUFFIXES.any? { |suffix| path.end_with?(suffix) }

          raise Gigatoken::Error, "#{path}: the CLI has no zstd decoder (the zstd-ruby gem is not a dependency); " \
            "decompress it first, or use Tokenizer#encode_files, which decompresses .zst natively"
        end

        def gzip?(path)
          path.end_with?(".gz")
        end

        def darwin_cpu_info
          name = sysctl("machdep.cpu.brand_string")
          [name, sysctl_int("hw.physicalcpu"), sysctl_int("hw.packages")]
        end

        def sysctl(key)
          output = IO.popen(["sysctl", "-n", key], err: File::NULL, &:read)
          output.strip unless output.nil? || output.empty? || !$?.success?
        rescue Errno::ENOENT
          nil
        end

        def sysctl_int(key)
          value = sysctl(key)
          Integer(value) if value&.match?(/\A\d+\z/)
        end

        # Within each processor block "physical id" precedes "core id", so
        # (socket, core) pairs count physical cores across sockets.
        def linux_cpu_info
          name = nil
          physical_ids = Set.new
          socket_core_ids = Set.new
          physical_id = ""
          File.foreach("/proc/cpuinfo") do |line|
            key, _, value = line.partition(":")
            key, value = key.strip, value.strip
            case key
            when "model name" then name ||= value
            when "physical id"
              physical_id = value
              physical_ids << value
            when "core id" then socket_core_ids << [physical_id, value]
            end
          end
          [name, (socket_core_ids.size unless socket_core_ids.empty?), (physical_ids.size unless physical_ids.empty?)]
        rescue Errno::ENOENT
          [nil, nil, nil]
        end
      end
    end
  end
end
