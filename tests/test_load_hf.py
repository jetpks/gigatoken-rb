import pytest
from pytest import fixture, param

from gigatok._load.hf import load_hf_tokenizer


@pytest.mark.parametrize("name_or_path", ["openai-community/gpt2", "Qwen/Qwen2-1.5B-Instruct"])
def test_load_hf_tokenizer(name_or_path: str):
    tokenizer = load_hf_tokenizer(name_or_path)
    print(tokenizer)
    assert tokenizer is not None
