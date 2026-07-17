"""Drop gigatoken into code written for tiktoken.

`as_tiktoken()` wraps a Tokenizer in a drop-in for the `tiktoken.Encoding`
API. Run with: uv run examples/tiktoken_drop_in.py
"""

import gigatoken as gt

# Any Tokenizer source works; a .tiktoken vocabulary file loads directly
# with gt.Tokenizer.from_tiktoken("r50k_base.tiktoken").
encoding = gt.Tokenizer("openai-community/gpt2").as_tiktoken()

ids = encoding.encode("Tokenize your text data at GB/s!")
print(ids)
print(encoding.decode(ids))

# Special-token handling matches tiktoken: disallowed by default,
# encoded as specials with allowed_special="all".
text = "A document.<|endoftext|>"
ids = encoding.encode(text, allowed_special="all")
assert ids[-1] == encoding.eot_token
assert encoding.decode(ids) == text

print(encoding.encode_batch(["one doc", "another doc"]))
print(f"{encoding.n_vocab=}")
