"""Drop gigatoken into code written for HuggingFace transformers.

`as_hf()` wraps a Tokenizer in a drop-in for the fast-tokenizer API with
identical output. Run with: uv run examples/hf_drop_in.py
"""

import numpy as np
from transformers import AutoTokenizer

import gigatoken as gt

hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
tokenizer = gt.Tokenizer(hf_tokenizer).as_hf()

# Use it exactly where the transformers tokenizer was used before.
docs = ["This is a test string", "Here is another"]
expected = hf_tokenizer(docs, return_tensors="np", padding=True)
actual = tokenizer(docs, return_tensors="np", padding=True)
np.testing.assert_equal(actual.input_ids, expected.input_ids)
np.testing.assert_equal(actual.attention_mask, expected.attention_mask)
print(actual.input_ids)

# The usual helpers behave the same way.
assert tokenizer.encode("hello world") == hf_tokenizer.encode("hello world")
assert tokenizer.batch_decode(actual.input_ids, skip_special_tokens=True) == hf_tokenizer.batch_decode(expected.input_ids, skip_special_tokens=True)
print(tokenizer.batch_decode(actual.input_ids, skip_special_tokens=True))
