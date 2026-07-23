# frozen_string_literal: true

require "etc"

module Gigatoken
  module CLI
    # Helpers shared by the bench and validate commands: tokenizer loading,
    # byte-size parsing, document splitting, and CPU identification.
    module Support
      SIZE_UNITS = {"" => 1, "k" => 10**3, "m" => 10**6, "g" => 10**9, "t" => 10**12}.freeze
      private_constant :SIZE_UNITS

      SIZE_PATTERN = /\A\s*(\d+(?:\.\d+)?)\s*([kmgt]?)i?b?\s*\z/i
      private_constant :SIZE_PATTERN

      class << self
        # Load TOKENIZER: a tokenizer.json path/directory, a HuggingFace
        # repo id, or a .tiktoken file — see Gigatoken::Tokenizer.load.
        def load_tokenizer(spec)
          Gigatoken::Tokenizer.load(spec)
        end

        # Parse a decimal byte size like "100MB", "2.5GB", or "1000000";
        # "none"/"unlimited" means no limit.
        def parse_size(text)
          return nil if ["none", "unlimited"].include?(text.strip.downcase)

          match = SIZE_PATTERN.match(text)
          raise Gigatoken::Error, "cannot parse size #{text.inspect}; expected something like 100MB (or 'none')" unless match

          (match[1].to_f * SIZE_UNITS.fetch(match[2].downcase)).to_i
        end

        # A Native::TextFileSource for FILES, split on `separator` when
        # given.
        def text_file_source(files, separator)
          Gigatoken::Native::TextFileSource.new(files.map(&:to_s), separator: separator)
        end

        # Whole files as raw bytes, one document per file, or (with a
        # separator) the separator-split pieces of each file in order, empty
        # documents skipped.
        def split_docs(files, separator)
          raws = files.map { |file| File.binread(file.to_s) }
          return raws if separator.nil?

          sep = separator.b
          raws.flat_map { |raw| raw.split(sep).reject(&:empty?) }
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
