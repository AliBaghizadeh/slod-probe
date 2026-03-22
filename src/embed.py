"""Prototype-compatible embedding wrapper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.embed_text import embed_dataset, load_embeddings


def build_embeddings(
    data_path: str | Path,
    model_name: str,
    cache_dir: str | Path,
    text_column: str = "text",
    batch_size: int = 16,
):
    """Compute or load cached frozen embeddings for a dataset."""
    return embed_dataset(
        data_path=data_path,
        text_column=text_column,
        model_name=model_name,
        batch_size=batch_size,
        cache_dir=cache_dir,
        metadata_columns=["span_id", "paper_id", "weak_label", "section_name"],
        resume=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prototype-compatible embedding extraction wrapper")
    parser.add_argument("--input", required=True, help="Input CSV")
    parser.add_argument("--model", required=True, help="Embedding model name")
    parser.add_argument("--cache-dir", required=True, help="Cache directory")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    embeddings, metadata = build_embeddings(
        data_path=args.input,
        model_name=args.model,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
    )
    print(f"Embeddings shape: {embeddings.shape}")
    if isinstance(metadata, pd.DataFrame):
        print(f"Metadata rows: {len(metadata)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
