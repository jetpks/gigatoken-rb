# frozen_string_literal: true

require_relative "support"

module Gigatoken
  module CLI
    # `gigatoken bench TOKENIZER FILES...` — measure encode throughput in
    # MB/s and Mtok/s, mirroring the Python CLI's `gigatoken bench` output
    # shape (gigatoken/_cli.py).
    class Bench < Dry::CLI::Command
      desc "Measure the time to encode FILES with TOKENIZER"

      argument :tokenizer, required: true, desc: "tokenizer.json path or directory, HuggingFace repo id, or .tiktoken file"
      argument :files, type: :array, required: true, desc: "UTF-8 text files to encode"

      option :doc_separator, desc: 'document separator to split the files on, e.g. "<|endoftext|>"; whole files are single documents otherwise'
      option :limit_bytes, default: "100MB", desc: "cap the bytes benchmarked, e.g. 100MB; 'none' for everything"
      option :parallel, type: :boolean, default: true, desc: "encode on the worker pool instead of one document at a time"

      def call(tokenizer:, files:, doc_separator: nil, limit_bytes: "100MB", parallel: true, **)
        limit = Support.parse_size(limit_bytes)
        out.puts "#{label("cpu")}: #{Support.cpu_info}"

        gt_tokenizer = Support.load_tokenizer(tokenizer)
        docs = Support.subset_docs(Support.split_docs(files, doc_separator), limit)

        start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
        encoded = parallel ? gt_tokenizer.encode_batch(docs) : docs.map { |doc| gt_tokenizer.encode(doc) }
        seconds = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start

        out.puts report("gigatoken", seconds, docs.sum(&:bytesize), encoded.sum(&:length))
      rescue Gigatoken::Error => e
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
