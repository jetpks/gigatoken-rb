"""ParquetFileSource: encode_files / train_bpe over parquet columns.

Uses the Olmo3 tokenizer (no NFC normalization, so byte-exact roundtrips).
Fixture files are written with polars (dev dependency).
"""

import json

import pytest

from gigatoken import JsonlFileSource, ParquetFileSource, train_bpe
from gigatoken.gigatoken_rs import BPETokenizer

pl = pytest.importorskip("polars")

DOCS = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "Peter Piper picked a peck of pickled peppers.",
    "To be, or not to be, that is the question.",
    "All that glitters is not gold.",
    "A journey of a thousand miles begins with a single step.",
    "Once upon a time, there was a little girl named Lily.",
    "The sun was shining and the birds were singing.",
    "Tom and his friend went to the park to play.",
    "The stars came out at night and twinkled in the sky.",
] * 20


@pytest.fixture(scope="module")
def tok(olmo3_tokenizer_path) -> BPETokenizer:
    return BPETokenizer.from_hf(olmo3_tokenizer_path)


@pytest.fixture(scope="module")
def expected(tok):
    """Reference: per-document ids via single-doc encode."""
    return [tok.encode(d).tolist() for d in DOCS]


@pytest.fixture(scope="module")
def parquet_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("parquet") / "corpus.parquet"
    pl.DataFrame({"text": DOCS}).write_parquet(path)
    return str(path)


def _ids(out):
    return out.tolist()


# ---------------------------------------------------------------------------
# encode_files
# ---------------------------------------------------------------------------


def test_encode_files_parquet(tok, expected, parquet_file):
    assert _ids(tok.encode_files(ParquetFileSource([parquet_file]))) == expected


def test_encode_files_parquet_serial_matches(tok, expected, parquet_file):
    got = tok.encode_files(ParquetFileSource([parquet_file]), parallel=False)
    assert _ids(got) == expected


def test_encode_files_parquet_bare_path_defaults_to_text(tok, expected, parquet_file):
    """A bare .parquet path gets ParquetFileSource(column="text") semantics."""
    assert _ids(tok.encode_files(parquet_file)) == expected


def test_encode_files_parquet_multiple_row_groups(tok, expected, tmp_path):
    """Small row groups: read in parallel per group, row order preserved."""
    path = tmp_path / "grouped.parquet"
    pl.DataFrame({"text": DOCS}).write_parquet(path, row_group_size=7)
    assert _ids(tok.encode_files(ParquetFileSource([str(path)]))) == expected


def test_encode_files_parquet_custom_column(tok, expected, tmp_path):
    path = tmp_path / "custom.parquet"
    pl.DataFrame({"content": DOCS, "id": list(range(len(DOCS)))}).write_parquet(path)
    got = tok.encode_files(ParquetFileSource([str(path)], column="content"))
    assert _ids(got) == expected


def test_encode_files_parquet_multiple_files_in_order(tok, expected, tmp_path):
    paths = []
    for i, chunk in enumerate([DOCS[:70], DOCS[70:130], DOCS[130:]]):
        path = tmp_path / f"part{i}.parquet"
        pl.DataFrame({"text": chunk}).write_parquet(path)
        paths.append(str(path))
    assert _ids(tok.encode_files(ParquetFileSource(paths))) == expected


def test_encode_files_parquet_nulls_are_empty_docs(tok, tmp_path):
    """Null rows must stay in the output as empty documents (row-aligned)."""
    path = tmp_path / "nulls.parquet"
    docs = [DOCS[0], None, DOCS[1], None]
    pl.DataFrame({"text": docs}, schema={"text": pl.String}).write_parquet(path)
    got = _ids(tok.encode_files(ParquetFileSource([str(path)])))
    assert len(got) == len(docs)
    assert got[1] == [] and got[3] == []
    assert got[0] == tok.encode(DOCS[0]).tolist()
    assert got[2] == tok.encode(DOCS[1]).tolist()


def test_encode_files_parquet_missing_column_raises(tok, parquet_file):
    with pytest.raises(IOError, match="no column .content."):
        tok.encode_files(ParquetFileSource([parquet_file], column="content"))


def test_parquet_source_repr(parquet_file):
    source = ParquetFileSource([parquet_file], column="body")
    assert repr(source) == "ParquetFileSource(paths=[1 files], column=\"body\")"


# ---------------------------------------------------------------------------
# train_bpe
# ---------------------------------------------------------------------------

VOCAB_SIZE = 400


def test_train_parquet_matches_jsonl(tmp_path):
    """Training from parquet must produce identical merges to the same
    documents as JSONL."""
    jsonl_path = tmp_path / "corpus.jsonl"
    jsonl_path.write_text("".join(json.dumps({"text": d}) + "\n" for d in DOCS))
    parquet_path = tmp_path / "corpus.parquet"
    pl.DataFrame({"text": DOCS}).write_parquet(parquet_path, row_group_size=13)

    _, ref_merges = train_bpe(JsonlFileSource([str(jsonl_path)]), VOCAB_SIZE, [])
    vocab, merges = train_bpe(ParquetFileSource([str(parquet_path)]), VOCAB_SIZE, [])
    assert len(vocab) == VOCAB_SIZE
    assert merges == ref_merges

    # A bare .parquet path defaults to column "text" and must match too.
    _, bare_merges = train_bpe(str(parquet_path), VOCAB_SIZE, [])
    assert bare_merges == ref_merges
