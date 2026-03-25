import pytest
from pytest import fixture, param

from toker._load.hf import load_hf_config


@pytest.mark.parametrize("name_or_path", ["openai-community/gpt2", "Qwen/Qwen2-1.5B-Instruct"])
def test_load_hf_config(name_or_path: str):
    config = load_hf_config(name_or_path)
    assert config is not None
