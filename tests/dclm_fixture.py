"""Curated DCLM test corpus: ~20 MB of diverse, edge-case-heavy documents.

Downloads one pinned DCLM-baseline shard into the standard HuggingFace cache
(reused across runs and shared with other HF consumers — nothing is written
to the repo), then classifies each document into edge-case categories (CJK,
RTL/other scripts, NFC-divergent text, emoji, control whitespace, code, giant
unbroken tokens, ...) and fills a byte quota per category. The shard revision
and selection rules are fixed, so every machine selects the identical corpus.

Print selection stats with: uv run python tests/dclm_fixture.py
"""

import hashlib
import io
import json
import re
import sys
import unicodedata
from pathlib import Path

HF_REPO = "mlfoundations/dclm-baseline-1.0"
HF_REVISION = "a3b142c183aebe5af344955ae20836eb34dcf69b"
HF_SHARD = "global-shard_01_of_10/local-shard_0_of_10/shard_00000000_processed.jsonl.zst"

TARGET_BYTES = 20_000_000  # total selection size (UTF-8 bytes of text)
MAX_DOC_BYTES = 262_144  # skip anything larger outright

# Codepoint classes that make text hard for tokenizers.
_CJK_RE = re.compile(r"[　-ヿ㐀-䶿一-鿿가-힯豈-﫿＀-￯]")
_OTHER_SCRIPT_RE = re.compile(r"[Ͱ-ϿЀ-ӿ԰-֏֐-׿؀-ۿऀ-ॿ฀-๿]")
_EMOJI_RE = re.compile(r"[←-⇿☀-➿⬀-⯿\U0001f000-\U0001faff]")
_COMBINING_RE = re.compile(r"[̀-ͯ᪰-᫿⃐-⃿︠-︯]")
_ODD_WS_RE = re.compile(r"[\t\r\x0b\f\x85\xa0​-‍  　﻿]| {8,}|\n{4,}")
_LONG_TOKEN_RE = re.compile(r"\S{80,}")
_CODE_MARK_RE = re.compile(r"[{};]|\n[ \t]+\S|</\w|&\w+;")
_DIGIT_RE = re.compile(r"[0-9]")


def _count(pattern: re.Pattern, text: str, at_least: int) -> bool:
    """True if `pattern` matches at least `at_least` times in `text`."""
    n = 0
    for _ in pattern.finditer(text):
        n += 1
        if n >= at_least:
            return True
    return False


# (category, byte quota). A document lands in the first category it matches
# that still has quota left; order therefore puts the rarest classes first.
QUOTAS: list[tuple[str, int]] = [
    ("nfc_divergent", 1_500_000),
    ("other_script", 2_000_000),
    ("cjk", 2_000_000),
    ("combining", 1_000_000),
    ("emoji", 1_500_000),
    ("odd_whitespace", 2_000_000),
    ("long_token", 1_500_000),
    ("code", 2_500_000),
    ("digit_heavy", 1_500_000),
    ("huge", 1_500_000),
    ("tiny", 500_000),
    ("general", 2_500_000),
]


def _classify(text: str, nbytes: int) -> list[str]:
    """All categories a document qualifies for, in QUOTAS priority order."""
    cats = []
    if not text.isascii():
        if not unicodedata.is_normalized("NFC", text):
            cats.append("nfc_divergent")
        if _count(_OTHER_SCRIPT_RE, text, 50):
            cats.append("other_script")
        if _count(_CJK_RE, text, 50):
            cats.append("cjk")
        if _count(_COMBINING_RE, text, 10):
            cats.append("combining")
        if _count(_EMOJI_RE, text, 5):
            cats.append("emoji")
    if _count(_ODD_WS_RE, text, 5):
        cats.append("odd_whitespace")
    if _LONG_TOKEN_RE.search(text):
        cats.append("long_token")
    if _count(_CODE_MARK_RE, text, max(20, nbytes // 200)):
        cats.append("code")
    if len(_DIGIT_RE.findall(text)) * 6 > len(text):
        cats.append("digit_heavy")
    if nbytes >= 131_072:
        cats.append("huge")
    if nbytes < 200:
        cats.append("tiny")
    cats.append("general")
    return cats


def download_shard() -> Path:
    """Fetch the pinned DCLM shard into the HuggingFace cache (no-op when cached)."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=HF_REPO, filename=HF_SHARD, repo_type="dataset", revision=HF_REVISION))


def _iter_shard_texts(shard: Path):
    import zstandard

    with open(shard, "rb") as fh:
        reader = zstandard.ZstdDecompressor().stream_reader(fh)
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            if line.strip():
                text = json.loads(line).get("text")
                if text:
                    yield text


def select_dclm_docs(log=lambda msg: None) -> list[tuple[str, str]]:
    """Scan the shard and return the curated (category, text) selection."""
    shard = download_shard()
    remaining = dict(QUOTAS)
    selected: list[tuple[str, str]] = []
    spill: list[str] = []  # overflow docs kept to top up unfilled quotas
    spill_bytes = 0
    seen: set[bytes] = set()
    scanned = 0
    total = 0

    for text in _iter_shard_texts(shard):
        raw = text.encode("utf-8")
        nbytes = len(raw)
        scanned += nbytes
        if total >= TARGET_BYTES:
            break
        if nbytes > MAX_DOC_BYTES:
            continue
        h = hashlib.blake2b(raw, digest_size=16).digest()
        if h in seen:
            continue
        seen.add(h)
        for cat in _classify(text, nbytes):
            if remaining[cat] > 0:
                selected.append((cat, text))
                remaining[cat] -= nbytes
                total += nbytes
                break
        else:
            if spill_bytes < 8_000_000:
                spill.append(text)
                spill_bytes += nbytes

    # Quotas the shard couldn't fill are topped up from ordinary overflow
    # docs so the corpus still reaches ~20 MB.
    for text in spill:
        if total >= TARGET_BYTES:
            break
        selected.append(("general", text))
        total += len(text.encode("utf-8"))

    log(f"selected {len(selected)} docs, {total / 1e6:.1f} MB (scanned {scanned / 1e6:.0f} MB)")
    for cat, quota in QUOTAS:
        log(f"  {cat}: {sum(1 for c, _ in selected if c == cat)} docs, {(quota - remaining[cat]) / 1e6:.2f}/{quota / 1e6:.1f} MB")
    return selected


_DOCS: list[str] | None = None


def get_dclm_docs() -> list[str]:
    """The curated document texts, selected once per process."""
    global _DOCS
    if _DOCS is None:
        _DOCS = [text for _, text in select_dclm_docs(log=lambda msg: print(f"[dclm_fixture] {msg}", file=sys.stderr))]
    return _DOCS


if __name__ == "__main__":
    import time

    start = time.perf_counter()
    select_dclm_docs(log=print)
    print(f"took {time.perf_counter() - start:.1f}s")
