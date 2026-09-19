# gigatoken-rb documentation

Documentation for the `gigatoken` gem, organized by [Diátaxis](https://diataxis.fr/).

## Tutorials — learn by doing

- [Getting started](tutorials/getting-started.md) — install, load a packaged encoding, encode and decode, your first batch.

## How-to guides — task recipes

- [Load a tokenizer from any source](how-to/load-a-tokenizer.md)
- [Tokenize files without leaving Rust](how-to/tokenize-files.md)
- [Work with packed results](how-to/use-packed-results.md)
- [Tune the encode-cache budget](how-to/tune-the-cache-budget.md)
- [Run encodes under Async](how-to/run-under-async.md)
- [Measure time and allocations](how-to/measure-time-and-allocations.md)

## Reference — exact behaviour

- [`Gigatoken::Tokenizer`](reference/tokenizer.md)
- [`Gigatoken::PackedResult`](reference/packed-result.md)
- [File sources](reference/file-sources.md)
- [Encodings, settings and errors](reference/encodings-and-settings.md)
- [Command line](reference/cli.md)

## Explanation — concepts and design

- [Benchmarks: methodology and full results](explanation/benchmarks.md)
- [Allocations: what a call costs, and why the seam is shaped this way](explanation/allocations.md)
- [Async design](explanation/async-design.md)
