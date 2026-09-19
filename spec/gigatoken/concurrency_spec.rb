# frozen_string_literal: true

require_relative "../spec_helper"

# Ruby hands one tokenizer instance to every thread, so the wrapped state has to
# be safe under concurrent use. Before the RwLock, `encode` took a mutable
# borrow of a `RefCell` that the batch paths held shared across a GVL release —
# so a batch encode racing a single encode aborted the VM with "RefCell already
# borrowed", a *fatal*, not a rescuable exception. The RwLock that replaced it
# then had two failure modes of its own, both fixed here: a reader blocking
# behind a queued writer *with the GVL held* deadlocked the whole VM, and an
# interrupt arriving during a batch longjmped out of `rb_nogvl` over every Rust
# frame, leaving the lock held and the input Strings locked forever.
#
# Each example runs in a subprocess, with an external kill timeout: a regression
# here hangs or kills the interpreter, and that would take the whole suite with
# it rather than reporting one red example.
RSpec.describe "concurrent use of a shared tokenizer" do
  def run_ruby(source, timeout: 60)
    lib = File.expand_path("../../lib", __dir__)
    io = IO.popen([RbConfig.ruby, "-I", lib, "-e", source], err: [:child, :out])
    reader = Thread.new { io.read }
    Process.kill("KILL", io.pid) unless reader.join(timeout)
    out = reader.value
    io.close
    [$CHILD_STATUS || $?, out]
  end

  let(:preamble) { <<~RUBY }
    Warning[:experimental] = false
    require "gigatoken"
    tok = Gigatoken::Tokenizer.from_encoding("cl100k_base")
    corpus = (1..32).map { |i| "document \#{i} " + ("lorem ipsum dolor " * (i % 11 + 2)) }
  RUBY

  # The reviews' corpus: random words, so the pretoken cache never makes a
  # batch cheap and there is a real window in which to interrupt one.
  let(:uncached) { <<~RUBY }
    def uncached_corpus(docs, words)
      srand(7)
      Array.new(docs) { Array.new(words) { (0...(4 + rand(6))).map { (97 + rand(26)).chr }.join }.join(" ") }
    end
  RUBY

  it "survives a batch encode racing single encodes on the same instance" do
    status, out = run_ruby(<<~RUBY)
      #{preamble}
      expected = tok.encode(corpus.first)
      big = 6.times.map { corpus.join(" ") * 40 }

      batcher = Thread.new { 40.times { tok.encode_batch(big) } }
      singles = Thread.new { 4000.times { raise "wrong" unless tok.encode(corpus.first) == expected } }
      [batcher, singles].each(&:join)
      puts "OK"
    RUBY

    expect(out).to include("OK"), "subprocess died: #{out}"
    expect(status).to be_success
  end

  it "returns identical output for the same input across many threads" do
    status, out = run_ruby(<<~RUBY)
      #{preamble}
      expected = corpus.map { |s| tok.encode(s) }

      results = 8.times.map do
        Thread.new { 200.times.flat_map { corpus.map { |s| tok.encode(s) } } }
      end.map(&:value)

      results.each do |r|
        r.each_slice(corpus.size) { |slice| raise "mismatch" unless slice == expected }
      end
      puts "OK"
    RUBY

    expect(out).to include("OK"), "subprocess died: #{out}"
    expect(status).to be_success
  end

  it "keeps decode and the vocab readers usable while a batch encode runs" do
    status, out = run_ruby(<<~RUBY)
      #{preamble}
      big = 6.times.map { corpus.join(" ") * 40 }
      expected_ids = tok.encode(corpus.first)

      batcher = Thread.new { 30.times { tok.encode_batch(big) } }
      readers = Thread.new do
        2000.times do
          raise "decode" unless tok.decode(expected_ids).is_a?(String)
          raise "vocab_size" unless tok.vocab_size.positive?
          raise "cache_entries" unless tok.cache_entries.is_a?(Integer)
        end
      end
      [batcher, readers].each(&:join)
      puts "OK"
    RUBY

    expect(out).to include("OK"), "subprocess died: #{out}"
    expect(status).to be_success
  end

  # A batch in flight, an `encode` queued behind it, and a third thread's
  # readers. Under the writer-preferring `RwLock` this type used to hold, the
  # reader blocked behind the queued writer *while holding the GVL*, so the
  # batch could never reacquire the GVL to drop its guard: the whole VM hung,
  # SIGKILL-only, `Thread#join` and signal handlers included.
  it "completes a batch, an encode queued behind it and a third thread's readers" do
    status, out = run_ruby(<<~RUBY)
      #{preamble}
      big = 8.times.map { corpus.join(" ") * 40 }
      expected = tok.encode(corpus.first)

      batcher = Thread.new { 40.times { tok.encode_batch(big, packed: true) } }
      writer = Thread.new { 200.times { raise "wrong" unless tok.encode(corpus.first) == expected } }
      readers = Thread.new do
        2000.times do
          raise "decode" unless tok.decode(expected).is_a?(String)
          raise "vocab_size" unless tok.vocab_size.positive?
          raise "cache_entries" unless tok.cache_entries.is_a?(Integer)
        end
      end
      [batcher, writer, readers].each(&:join)
      puts "OK"
    RUBY

    expect(out).to include("OK"), "subprocess died or hung: #{out}"
    expect(status).to be_success
  end

  # An interrupt during a batch used to be raised by `rb_nogvl` itself,
  # longjmping over every Rust frame: the read guard was never dropped and
  # every input String kept its `STR_TMPLOCK`, so mutating one raised and the
  # next `encode` hung forever. It must now cancel the batch and unwind
  # normally, leaving inputs and tokenizer exactly as they were.
  it "cancels the batch a Timeout interrupts, leaving the inputs mutable" do
    status, out = run_ruby(<<~RUBY)
      #{preamble}
      #{uncached}
      require "timeout"
      def clock = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      docs = uncached_corpus(3000, 1500)
      full = 2.times.map { t = clock; tok.encode_batch(docs, packed: true); clock - t }.min

      t0 = clock
      begin
        Timeout.timeout(full / 4) { tok.encode_batch(docs, packed: true) }
        raise "no Timeout::Error raised"
      rescue Timeout::Error
      end
      took = clock - t0
      raise "batch ran on past the timeout (\#{took.round(3)}s of \#{full.round(3)}s)" if took > full * 0.75

      docs[0] << " still mutable"
      raise "encode after the cancelled batch" unless tok.encode(corpus.first) == tok.encode(corpus.first.dup)
      puts "OK"
    RUBY

    expect(out).to include("OK"), "subprocess died or hung: #{out}"
    expect(status).to be_success
  end

  it "leaves the inputs mutable and the tokenizer usable after Thread#kill" do
    status, out = run_ruby(<<~RUBY)
      #{preamble}
      #{uncached}
      docs = uncached_corpus(1500, 800)
      batcher = Thread.new { 50.times { tok.encode_batch(docs) } }
      sleep 0.05
      batcher.kill
      batcher.join

      docs[0] << " still mutable"
      raise "encode after the killed batch" unless tok.encode(corpus.first).is_a?(Array)
      puts "OK"
    RUBY

    expect(out).to include("OK"), "subprocess died or hung: #{out}"
    expect(status).to be_success
  end

  # The same interrupt under a fiber scheduler with a worker pool, where the
  # encode runs on an `IO::Event::WorkerPool` thread and the timeout reaches
  # it through `rb_fiber_scheduler_blocking_operation_cancel` — the same
  # unblock function, a different caller.
  it "cancels the batch an Async timeout interrupts" do
    status, out = run_ruby(<<~RUBY)
      ENV["ASYNC_SCHEDULER_WORKER_POOL"] = "true"
      #{preamble}
      #{uncached}
      require "async"
      docs = uncached_corpus(1500, 800)

      timed_out = false
      Async do |task|
        task.with_timeout(0.05) { 50.times { tok.encode_batch(docs, packed: true) } }
      rescue Async::TimeoutError
        timed_out = true
      end
      raise "no Async::TimeoutError raised" unless timed_out

      docs[0] << " still mutable"
      raise "encode after the cancelled batch" unless tok.encode(corpus.first).is_a?(Array)
      puts "OK"
    RUBY

    expect(out).to include("OK"), "subprocess died or hung: #{out}"
    expect(status).to be_success
  end

  # The SentencePiece half of the fix: `SentencePieceTokenizer`'s model is a
  # plain field (every path only reads it) and its one mutable piece —
  # `EncodeState`, the pretoken cache — is a `Mutex` rather than the `RefCell`
  # it used to be. These two examples are guards, not discriminators: the
  # architect has confirmed both pass on the pre-0.2.1 tree (a1e7caa) too,
  # since the SP model was never `borrow_mut`'d there and `state.borrow_mut()`
  # is only ever reached under the GVL — no SP crash was reachable before this
  # change. They stay here as regression coverage for the `Mutex` conversion,
  # not as before/after proof.
  sp_fixture_path = File.expand_path("../fixtures/sp_tokenizer.json", __dir__)

  let(:sp_preamble) { <<~RUBY }
    Warning[:experimental] = false
    require "gigatoken"
    tok = Gigatoken::Tokenizer.from_file(#{sp_fixture_path.inspect})
    corpus = ["hello world", "hello", "world", "\u{1F389}", "hello world " * 8]
  RUBY

  it "survives an SP batch encode racing single encodes on the same instance" do
    status, out = run_ruby(<<~RUBY)
      #{sp_preamble}
      expected = tok.encode(corpus.first)
      big = corpus * 40

      batcher = Thread.new { 200.times { tok.encode_batch(big) } }
      singles = Thread.new { 4000.times { raise "wrong" unless tok.encode(corpus.first) == expected } }
      [batcher, singles].each(&:join)
      puts "OK"
    RUBY

    expect(out).to include("OK"), "subprocess died: #{out}"
    expect(status).to be_success
  end

  it "returns identical output for the same SP input across many threads" do
    status, out = run_ruby(<<~RUBY)
      #{sp_preamble}
      expected = corpus.map { |s| tok.encode(s) }

      results = 8.times.map do
        Thread.new { 200.times.flat_map { corpus.map { |s| tok.encode(s) } } }
      end.map(&:value)

      results.each do |r|
        r.each_slice(corpus.size) { |slice| raise "mismatch" unless slice == expected }
      end
      puts "OK"
    RUBY

    expect(out).to include("OK"), "subprocess died: #{out}"
    expect(status).to be_success
  end

  # mimalloc#1287: the extension's global allocator (mimalloc) used to tear
  # down its static main heap when the thread that first `require`d it exited,
  # so a later thread reusing that pthread_t segfaulted in `mi_thread_init` on
  # its first encode. Falcon `--threaded` hits this on instance restarts: the
  # loader thread exits, and the next request thread crashes the interpreter.
  it "survives encoding from fresh threads after the loading thread exits" do
    status, out = run_ruby(<<~RUBY)
      Warning[:experimental] = false
      loader = Thread.new do
        require "gigatoken"
        Gigatoken::Tokenizer.from_encoding("o200k_base")
      end
      tok = loader.value
      100.times { |i| Thread.new { tok.encode("hello again \#{i}") }.join }
      puts "OK"
    RUBY

    expect(out).to include("OK"), "subprocess died: #{out}"
    expect(status).to be_success
  end
end
