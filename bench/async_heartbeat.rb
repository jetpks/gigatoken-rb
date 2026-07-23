#!/usr/bin/env ruby
# frozen_string_literal: true

# Observable proof (I09) that Tokenizer#encode_batch cooperates with the
# Async reactor only when Fiber.scheduler has a worker pool. Run twice:
#
#   ASYNC_SCHEDULER_WORKER_POOL=true ruby -Ilib bench/async_heartbeat.rb
#   ruby -Ilib bench/async_heartbeat.rb
#
# A heartbeat fiber ticks every ~1ms alongside one encode_batch call; if the
# reactor keeps ticking while the encode runs, the calling fiber yielded.

require "async"
require "gigatoken"

FIXTURE = File.expand_path("../tests/fixtures/gpt2_tokenizer.json", __dir__)
tokenizer = Gigatoken::Tokenizer.from_file(FIXTURE)

text = File.read(FIXTURE)
# 320 copies of the ~1.3MB fixture (~430MB) pushes encode_batch's wall time
# to ~1s on this machine, comfortably past the >= 0.5s target.
batch = Array.new(320) { text }

ticks = 0

Async do |task|
  heartbeat = task.async do
    loop do
      ticks += 1
      sleep(0.001)
    end
  end

  tokenizer.encode_batch(batch)

  heartbeat.stop
end

puts "TICKS_DURING_ENCODE=#{ticks}"
puts "ASYNC_YIELD: #{(ticks >= 5) ? "yes" : "no"}"
