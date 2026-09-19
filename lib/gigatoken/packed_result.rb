# frozen_string_literal: true

module Gigatoken
  # A packed batch-encode result: one `IO::Buffer` of u32 token ids (native
  # byte order) for every document, plus `lens`, each document's token count.
  # Returned by `Tokenizer#encode_batch`/`#encode_files` with `packed: true`
  # — the Ruby analog of the numpy/awkward-array path the native ext's
  # header comment says was dropped for lack of a Ruby equivalent. Documents
  # are unpacked from the buffer on demand via `#[]`/`#each`.
  class PackedResult
    include Enumerable

    attr_reader :buffer, :lens

    def initialize(buffer, lens)
      @buffer = buffer
      @lens = lens
      @offsets = lens.inject([0]) { |offsets, len| offsets << offsets.last + len }
    end

    # Number of documents.
    def size
      lens.size
    end

    # Total token count across every document.
    def token_count
      lens.sum
    end

    # Array of token ids for document `i`, materialized on demand; negative
    # indices count from the end, out-of-range ones give nil, and a
    # non-Integer index is a TypeError, all like Array.
    def [](index)
      i = Integer.try_convert(index)
      raise TypeError, "no implicit conversion of #{index.class} into Integer" if i.nil?

      i += size if i.negative?
      return unless i >= 0 && i < size

      buffer.values(:u32, @offsets[i] * 4, lens[i])
    end

    def each
      return enum_for(:each) unless block_given?

      lens.each_index { |i| yield self[i] }
    end

    # A ragged Array of Arrays, one per document — the same shape
    # `encode_batch`/`encode_files` return with `packed: false`.
    def to_a
      Array.new(size) { |i| self[i] }
    end
  end
end
