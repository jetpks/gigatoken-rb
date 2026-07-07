from pathlib import Path

from gigatok import train_bpe

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

if __name__ == "__main__":
    train_bpe(DATA_DIR / "TinyStoriesV2-GPT4-train.txt", 10_000, [])
