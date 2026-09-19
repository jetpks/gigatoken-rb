# frozen_string_literal: true

require_relative "support"

module Gigatoken
  module CLI
    # `gigatoken bench TOKENIZER FILES...` — measure encode throughput in
    # MB/s and Mtok/s, mirroring upstream's Python CLI `gigatoken bench`
    # output shape.
    class Bench < Dry::CLI::Command
      desc "Measure the time to encode FILES with TOKENIZER"

      argument :tokenizer, required: true, desc: "tokenizer.json path or directory, packaged encoding name (e.g. \"cl100k_base\"), HuggingFace repo id, or .tiktoken file (requires --pretokenizer)"
      argument :files, type: :array, required: true, desc: "UTF-8 text files to encode"

      option :doc_separator, desc: 'document separator to split the files on, e.g. "<|endoftext|>"; whole files are single documents otherwise'
      option :limit_bytes, default: "none", desc: "cap the bytes benchmarked, e.g. 100MB; 'none' for everything (parallel mode only — ignored with --no-parallel)"
      option :parallel, type: :boolean, default: true, desc: "encode on the worker pool instead of the fused serial core path"
      option :packed, type: :boolean, default: false, desc: "time the fused native file path with a packed IO::Buffer result instead of per-document Ruby arrays (ignores --limit-bytes)"
      option :pretokenizer, desc: "pretokenizer scheme, required when TOKENIZER is a .tiktoken file (one of #{Native.pretokenizer_names.join(", ")}); ignored otherwise"

      def call(tokenizer:, files:, doc_separator: nil, limit_bytes: "none", parallel: true, packed: false, pretokenizer: nil, **)
        Support.check_usage!(files, doc_separator)
        limit = Support.parse_size(limit_bytes)
        out.puts "#{label("cpu")}: #{Support.cpu_info}"

        gt_tokenizer = Support.load_tokenizer(tokenizer, pretokenizer: pretokenizer)

        start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
        if packed
          encoded = gt_tokenizer.encode_files(Support.text_file_source(files, doc_separator), parallel: parallel, packed: true)
          n_tokens = encoded.token_count
        elsif parallel
          docs = Support.subset_docs(Support.split_docs(files, doc_separator), limit)
          encoded = gt_tokenizer.encode_batch(docs)
          n_tokens = encoded.sum(&:length)
        else
          encoded = gt_tokenizer.encode_files(Support.text_file_source(files, doc_separator), parallel: false)
          n_tokens = encoded.sum(&:length)
        end
        seconds = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start

        # The throughput is over the bytes encoded, which for a compressed
        # file are its decompressed bytes. Counting those means reading the
        # file, so the native paths count off the clock; the batch path
        # already holds the documents it encoded.
        n_bytes = docs ? docs.sum(&:bytesize) : Support.input_bytesize(files)
        out.puts report("gigatoken", seconds, n_bytes, n_tokens)
      rescue Gigatoken::Error, SystemCallError => e
        err.puts "error: #{e.message}"
        exit(1)
      end

      private

      def label(name)
        format("%9s", name)
      end

      def report(name, seconds, n_bytes, n_tokens)
        mb = n_bytes / 1e6
        mtok = n_tokens / 1e6
        format("%s: %8.3f s | %10.2f MB at %8.2f MB/s | %8.2f Mtok at %7.2f Mtok/s",
          label(name), seconds, mb, mb / seconds, mtok, mtok / seconds)
      end
    end
  end
end
