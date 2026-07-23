# frozen_string_literal: true

require_relative "support"

module Gigatoken
  module CLI
    # `gigatoken validate TOKENIZER FILES...` — a Ruby-vs-Ruby consistency
    # check: encode FILES via `Tokenizer#encode_files` (native file loading
    # and splitting) and via a Ruby-side split plus `Tokenizer#encode_batch`,
    # and confirm the two paths agree per document. Cross-library validation
    # against another tokenizer implementation is out of scope for v1
    # (BRIEF §3.4).
    class Validate < Dry::CLI::Command
      desc "Check that encode_files agrees with a Ruby-side split plus encode_batch on FILES"

      argument :tokenizer, required: true, desc: "tokenizer.json path or directory, HuggingFace repo id, or .tiktoken file"
      argument :files, type: :array, required: true, desc: "UTF-8 text files to encode"

      option :doc_separator, desc: 'document separator to split the files on, e.g. "<|endoftext|>"; whole files are single documents otherwise'

      def call(tokenizer:, files:, doc_separator: nil, **)
        gt_tokenizer = Support.load_tokenizer(tokenizer)

        via_files = gt_tokenizer.encode_files(Support.text_file_source(files, doc_separator))
        via_batch = gt_tokenizer.encode_batch(Support.split_docs(files, doc_separator))

        if via_files.length != via_batch.length
          err.puts "validation FAILED: encode_files produced #{via_files.length} documents, encode_batch produced #{via_batch.length}"
          exit(1)
        end

        via_files.each_with_index do |doc, index|
          next if doc == via_batch[index]

          report_mismatch(index, doc, via_batch[index])
          exit(1)
        end

        out.puts "validation OK: #{via_files.length} documents match"
      rescue Gigatoken::Error => e
        err.puts "error: #{e.message}"
        exit(1)
      end

      private

      def report_mismatch(index, from_files, from_batch)
        at = from_files.zip(from_batch).index { |a, b| a != b } || [from_files.length, from_batch.length].min
        err.puts "validation FAILED: document #{index}: first mismatch at token #{at} " \
          "(encode_files #{from_files[at, 5]}... vs encode_batch #{from_batch[at, 5]}..., " \
          "lengths #{from_files.length} vs #{from_batch.length})"
      end
    end
  end
end
