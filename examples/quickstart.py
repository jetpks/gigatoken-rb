"""Load a tokenizer and encode/decode text with the gigatoken API.

Run with: uv run examples/quickstart.py
"""

import awkward as ak

import gigatoken as gt

# Accepts a HuggingFace Hub repo id, a path to a tokenizer.json (or a
# directory containing one), or an already-initialized HF tokenizer.
tokenizer = gt.Tokenizer("openai-community/gpt2")
print(f"loaded {tokenizer!r} with vocab_size={tokenizer.vocab_size}")

# Single documents: encode returns a numpy uint32 array, decode returns bytes.
ids = tokenizer.encode("Tokenize your text data at GB/s!")
print(ids)
print(tokenizer.decode(ids).decode("utf-8"))

# Batches encode in parallel and return a ragged awkward Array,
# one row of token ids per document.
docs = ["The first document.", "And a second, slightly longer one."]
tokens = tokenizer.encode_batch(docs)
print(tokens)
print(ak.num(tokens))  # tokens per document

# For model input, assemble a padded (rows x width) matrix directly in Rust,
# along with each row's real length.
matrix, lengths = tokenizer.encode_batch_padded(docs, pad_id=0)
print(matrix)
print(lengths)
