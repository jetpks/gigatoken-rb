from pathlib import Path

import tokenizers
from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# tokenizer = Tokenizer.from_file("tests/scripts/tokenizer_weighted_128k.json")

# print(tokenizer.encode("Hello, world! This is a test. Let's see how it works."))

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(str(Path("tests/scripts/unbox_tokenizer").resolve()))

    encoded = tokenizer.encode("Hello, world! This is a test. Let's see how it works.")
    decoded = tokenizer.decode(encoded)
    print(decoded)
    print(encoded)

    print(tokenizer)
