# from tiktoken.load import load_tiktoken_bpe
import tiktoken
from pytest import fixture


@fixture
def tiktoken_r50k():
    return tiktoken.get_encoding("r50k_base")


@fixture
def toker_r50k():
    from toker.toker_rs import BPETokenizer

    return BPETokenizer.from_tiktoken("/Users/marcel/data/tokenizers/r50k_base.tiktoken")


# def test_use_tiktoken_model(tiktoken_r50k):
#     print(t)


def test_use_toker_model(toker_r50k):
    print(toker_r50k)
    print(toker_r50k.encode(b"Here's a test string"))
