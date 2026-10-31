"""
Download the IMDB movie review dataset from Hugging Face and save it in the
original aclImdb folder layout expected by the notebook:

    aclImdb/
    ├── train/
    │   ├── pos/   (12,500 reviews)
    │   └── neg/   (12,500 reviews)
    └── test/
        ├── pos/   (12,500 reviews)
        └── neg/   (12,500 reviews)

Usage (from the repository root):
    python scripts/download_data.py
"""

from pathlib import Path

from datasets import load_dataset


def main(base_path: Path = Path("aclImdb")) -> None:
    # 1. Download dataset from Hugging Face
    dataset = load_dataset("stanfordnlp/imdb")

    # 2. Create folders and save each review as a .txt file
    for split in ["train", "test"]:
        for label, sentiment in [(1, "pos"), (0, "neg")]:
            reviews = [row["text"] for row in dataset[split] if row["label"] == label]

            folder = base_path / split / sentiment
            folder.mkdir(parents=True, exist_ok=True)

            for i, review in enumerate(reviews):
                (folder / f"{i}_{label}.txt").write_text(review, encoding="utf-8")

            print(f"{split}/{sentiment}: {len(reviews)} reviews saved")

    print("\nConversion completed.")
    print("Folder created at:", base_path.resolve())


if __name__ == "__main__":
    main()
