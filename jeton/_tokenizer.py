"""Unified high-level Tokenizer wrapping the Rust backends."""

from __future__ import annotations

from jeton._load.hf import to_tokenizer_json
from jeton.jeton_rs import BPETokenizer, SentencePieceTokenizer, load_hf_json

_BACKEND_TYPES = (BPETokenizer, SentencePieceTokenizer)


class Tokenizer:
    """A tokenizer in one of the standard formats supported by the library.

    Construct it from a path to a HuggingFace tokenizer.json (or a directory
    containing one), from an already-initialized HuggingFace tokenizer (a
    `tokenizers.Tokenizer` or a `transformers` tokenizer, fast or slow), or
    from an existing Rust backend instance. The right backend — byte-level
    BPE or SentencePiece BPE with byte fallback — is chosen automatically
    from the tokenizer's configuration.
    """

    def __init__(self, tokenizer):
        if isinstance(tokenizer, Tokenizer):
            self._backend = tokenizer._backend
        elif isinstance(tokenizer, _BACKEND_TYPES):
            self._backend = tokenizer
        else:
            self._backend = load_hf_json(to_tokenizer_json(tokenizer))

    @classmethod
    def from_json(cls, data: str | bytes) -> "Tokenizer":
        """Load from in-memory tokenizer.json contents."""
        return cls(load_hf_json(data))

    @classmethod
    def from_tiktoken(cls, path) -> "Tokenizer":
        """Load from a .tiktoken vocabulary file."""
        return cls(BPETokenizer.from_tiktoken(path))

    @property
    def backend(self):
        """The underlying Rust tokenizer (BPETokenizer or SentencePieceTokenizer)."""
        return self._backend

    def encode(self, input):
        return self._backend.encode(input)

    def encode_batch(self, inputs):
        return self._backend.encode_batch(inputs)

    def encode_files(self, source):
        return self._backend.encode_files(source)

    def decode(self, tokens) -> bytes:
        return self._backend.decode(tokens)

    def __getattr__(self, name):
        # Backend-specific extras (e.g. SentencePiece's encode_no_normalize).
        if name == "_backend":
            raise AttributeError(name)
        return getattr(self._backend, name)

    def __repr__(self) -> str:
        return f"Tokenizer({self._backend!r})"
