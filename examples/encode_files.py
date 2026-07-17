"""Encode documents straight from files — the fastest path.

Rust reads, splits, and encodes the files in parallel; no text passes
through Python. Run with: uv run examples/encode_files.py
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import awkward as ak

import gigatoken as gt

tokenizer = gt.Tokenizer("openai-community/gpt2")

with TemporaryDirectory() as tmp:
    # Plain text files: documents are the pieces between `separator`
    # occurrences. Without a separator, each whole file is one document.
    text_path = Path(tmp) / "corpus.txt"
    text_path.write_text("The first document.<|endoftext|>The second document.")
    tokens = tokenizer.encode_files(gt.TextFileSource([text_path], separator="<|endoftext|>"))
    print(tokens, ak.num(tokens))

    # JSON Lines files: one document per line, text taken from `field`.
    jsonl_path = Path(tmp) / "corpus.jsonl"
    jsonl_path.write_text("\n".join(json.dumps({"text": t}) for t in ["Document one.", "Document two."]))
    tokens = tokenizer.encode_files(gt.JsonlFileSource([jsonl_path], field="text"))
    print(ak.num(tokens))

    # A bare path (or list of paths) also works: one document per file.
    tokens = tokenizer.encode_files(text_path)

# For data already in memory, BytesSource is the buffer analog: pass whole
# buffers plus a separator and let Rust do the splitting — this is much
# faster than pre-splitting into a list of strings in Python.
data = "Doc A<|endoftext|>Doc B<|endoftext|>Doc C".encode()
tokens = tokenizer.encode_batch(gt.BytesSource(data, separator="<|endoftext|>"))
print(ak.num(tokens))
