---
type: how-to
---

# Measure time and allocations

Two scripts under `bench/` reproduce the numbers on the
[benchmarks page](../explanation/benchmarks.md). Both run against whatever
extension is currently compiled into `lib/gigatoken/`, so `bundle exec rake
compile` first after changing the Rust.

## Every operation, with tiktoken_ruby beside it

```sh
ruby -Ilib bench/operations.rb
BENCH_QUICK=1 ruby -Ilib bench/operations.rb    # smaller batches, shorter timing windows
```

Prints one Markdown table: iterations per second (`benchmark-ips`), Ruby
objects allocated per call (`GC.stat`, exact) and Ruby-heap bytes
malloc'd per call, for `encode` at three sizes, `decode`, ragged and packed
batches, packed-result access, `encode_files`, and construction.
`tiktoken_ruby` columns appear where it has an equivalent call.

Rust-side allocations don't show in Ruby's counters (the extension routes
them through mimalloc), so a change on that side shows up in the time
column, not the objects column.

## Compare two builds

Compile the other revision into a worktree and point `-I` at its `lib`:

```sh
git worktree add ../gigatoken-main main
(cd ../gigatoken-main && bundle exec rake compile)
ruby -I ../gigatoken-main/lib bench/operations.rb > main.md
ruby -Ilib bench/operations.rb > branch.md
```

Run them one after another, not concurrently: the batch rows use every core.

## Single-encode A/B with a noise floor

```sh
ruby -Ilib bench/encode_ab.rb
```

Times `encode` alone at three sizes, reports medians with a bootstrapped
A/A noise floor, and says whether a delta clears it. Its header comment
explains the design; the benchmarks page's 0.2.1 section shows it in use.

## Freeze a budget

`spec/gigatoken/allocations_spec.rb` asserts the objects-per-call each
operation may cost. When you remove an allocation, lower the budget there
so it can't come back.
