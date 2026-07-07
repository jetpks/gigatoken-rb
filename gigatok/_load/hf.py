"""Loading tokenizer configurations from HuggingFace sources.

Nothing here imports `transformers` or `tokenizers` at module level; those
packages are only touched when the caller hands us one of their objects (in
which case they are necessarily already installed).
"""

from __future__ import annotations

import os
from pathlib import Path


def load_hf_tokenizer(pretrained_model_name_or_path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
    return tokenizer


def to_tokenizer_json(source) -> str | bytes:
    """Resolve `source` to the contents of a HuggingFace tokenizer.json.

    Accepts a path to a tokenizer.json file (or a directory containing one),
    a `tokenizers.Tokenizer`, or a `transformers` tokenizer — fast ones
    (TokenizersBackend) through their backend, slow ones by converting with
    `transformers.convert_slow_tokenizer`.
    """
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        if path.is_dir():
            path = path / "tokenizer.json"
        return path.read_bytes()

    root_module = type(source).__module__.split(".")[0]

    # tokenizers.Tokenizer (or anything else that serializes itself the same way)
    to_str = getattr(source, "to_str", None)
    if callable(to_str) and root_module == "tokenizers":
        return to_str()

    # transformers fast tokenizer: backed by a tokenizers.Tokenizer
    backend = getattr(source, "backend_tokenizer", None)
    if backend is not None and callable(getattr(backend, "to_str", None)):
        return backend.to_str()

    # transformers slow tokenizer: convert to a tokenizers.Tokenizer first
    if root_module == "transformers":
        from transformers.convert_slow_tokenizer import convert_slow_tokenizer

        return convert_slow_tokenizer(source).to_str()

    raise TypeError(
        f"cannot extract a tokenizer.json from {type(source).__name__!r}; "
        "expected a path, a tokenizers.Tokenizer, or a transformers tokenizer"
    )
