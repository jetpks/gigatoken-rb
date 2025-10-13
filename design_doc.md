# Design Document for Toker

## Reading and parsing from document files

### Input data

Generic collection of document input processing supporting

* Collection of Bytes from Python (including generator)
* Collection of paths to files from Python

### Extracting documents from files

Each document can have various byte formats, that need to be parsed to separate into docs that can be pretokenized.

* Straight document bytes
* Separated by `<|endoftext|>` or similar
* JsonLines
* Compressed files (most importantly gzip compression)
* Parquet

Note that several of these can require allocating memory, since for instance json files must be transformed to contain
for instance newlines from '\' 'n' to '\n'.
Further, some formats are compressed, so must be uncompressed before they can be processed directly.
This means we can't entirely rely on references to memmapped bytes and must allow for ownership, although this is only
in certain cases.

Note that these should also support being passed along with document weights to bias pretoken counting.

## Output configuration for encoding

* Output to files similar to supported input formats
* List of numpy arrays

## Persistence of tokenizer

If a tokenizer is repeatedly reused from Python, we want to maintain the memory it has associated with caching.

## Configuring from Python

Example:

```python3
tokenizer_pipeline = (
    Tokenizer.from_tiktoken(tiktoken_name="r50k_base")
    .build_pipeline()
    .source_python_bytes()
    .sink_python_awkward()  # Allows for variable size output
)
tokenizer_pipeline.run(
    [b"This is some text", b"Some more bytes"],
)
```

Real structure:

```
class Tokenizer:
    ...
    def build_pipeline() -> PipelineBuildSource:
        ...

class PipelineBuildSource:
    def source_python_bytes() -> PipelineBuildSink[SourceKind]:
        ...

class PipelineBuildSink[SourceKind]:
    def sink_python_awkward() -> Pipeline[SourceKind, AwkwardArray]:
        ...

```

## Implementation Details

### Reading documents

Can read from:

- Bytes
- JsonLines

Challenge:

- There is often no "document" separation between docs in the Bytes setting.
  Instead, we have special tokens that break up the text.
  Want to output one stream of tokens, with the special tokens inserted in this stream.
  How can we preserve the special tokens we identified before pre-tokenization?

#### Parallel Strategies

| Input Type | Strategy                                                                                                        |
|:----------:|:----------------------------------------------------------------------------------------------------------------|
| JsonLines  | Find next newline, and perfectly split between docs at this boundary.                                           |
|   Bytes    | Find next sequence that is guaranteed to be split by the pretokenizer, for instance "\n." will always be split. |

Challenge:

- We can't always find any specific set of characters.
    - Maybe traverse a few steps using the pretokenizer iterator?
      Need to make sure invalid boundaries are removed in case of extremes.

### Ownership

When reading documents, we could read everything using a memmap, where borrowing from that memmap to every next step of
processing is fine.
Otherwise, we could have memory that only lives for the duration of a single document read, in which case we need to
save pretokens as they are processed.
Ideally, we hold on to them as references to the underlying read buffer until we've either tokenized or registered them
in a count hashmap.
