"""Length-control experiment for SLoD pipeline."""

import logging
from typing import Dict, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def truncate_text_to_tokens(text: str, target_tokens: int) -> str:
    """
    Truncate text to a fixed number of tokens.

    Uses simple whitespace-based tokenization.
    If text has fewer tokens than target, returns text unchanged.

    Args:
        text: input text.
        target_tokens: maximum number of tokens to keep.

    Returns:
        truncated text.
    """
    if not isinstance(text, str):
        return ""

    if target_tokens <= 0:
        return ""

    tokens = text.split()
    if len(tokens) <= target_tokens:
        return text

    truncated = " ".join(tokens[:target_tokens])
    return truncated


def apply_length_control(df: pd.DataFrame,
                         text_column: str = "text",
                         target_tokens: int = 100) -> pd.DataFrame:
    """
    Apply length control to a dataset by truncating text.

    Truncates all text to a fixed token length, updates token_count column.
    Preserves all other columns unchanged.

    Does NOT modify the original DataFrame.

    Args:
        df: input dataset with text column.
        text_column: name of text column.
        target_tokens: target token length (max tokens to keep).

    Returns:
        new DataFrame with truncated text and updated token_count.
    """
    df = df.copy()

    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in DataFrame")

    logger.info(f"Applying length control: target_tokens={target_tokens}")
    logger.info(f"Original token counts - mean: {df['token_count'].mean():.1f}, std: {df['token_count'].std():.1f}")

    # Truncate text
    df[text_column] = df[text_column].apply(
        lambda x: truncate_text_to_tokens(x, target_tokens)
    )

    # Update token_count
    df["token_count"] = df[text_column].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)

    logger.info(f"Controlled token counts - mean: {df['token_count'].mean():.1f}, std: {df['token_count'].std():.1f}")

    return df


def report_length_statistics(df_original: pd.DataFrame,
                              df_controlled: pd.DataFrame,
                              label_column: str = "weak_label") -> Dict:
    """
    Report statistics comparing original and controlled datasets.

    Reports:
    - class counts (original and controlled)
    - mean token count before control
    - mean token count after control
    - per-class token statistics

    Args:
        df_original: original dataset.
        df_controlled: length-controlled dataset.
        label_column: name of label column.

    Returns:
        dict with statistics.
    """
    stats = {}

    # Overall stats
    stats["n_samples"] = len(df_controlled)
    stats["n_papers"] = df_original["paper_id"].nunique() if "paper_id" in df_original.columns else "unknown"

    # Token statistics
    stats["original_token_stats"] = {
        "mean": float(df_original["token_count"].mean()),
        "std": float(df_original["token_count"].std()),
        "min": int(df_original["token_count"].min()),
        "max": int(df_original["token_count"].max()),
    }
    stats["controlled_token_stats"] = {
        "mean": float(df_controlled["token_count"].mean()),
        "std": float(df_controlled["token_count"].std()),
        "min": int(df_controlled["token_count"].min()),
        "max": int(df_controlled["token_count"].max()),
    }

    # Class counts
    class_counts_orig = df_original[label_column].value_counts(dropna=False).to_dict()
    class_counts_ctrl = df_controlled[label_column].value_counts(dropna=False).to_dict()

    stats["class_counts"] = {
        "original": {str(k): int(v) for k, v in class_counts_orig.items()},
        "controlled": {str(k): int(v) for k, v in class_counts_ctrl.items()},
    }

    # Per-class token statistics
    stats["per_class_token_stats"] = {}
    for label in df_original[label_column].unique():
        mask_orig = df_original[label_column] == label
        mask_ctrl = df_controlled[label_column] == label

        stats["per_class_token_stats"][str(label)] = {
            "original": {
                "mean": float(df_original[mask_orig]["token_count"].mean()),
                "count": int(mask_orig.sum()),
            },
            "controlled": {
                "mean": float(df_controlled[mask_ctrl]["token_count"].mean()),
                "count": int(mask_ctrl.sum()),
            },
        }

    return stats


def save_controlled_dataset(df: pd.DataFrame,
                            output_path: str | Path = "data/processed/spans_length_controlled.csv") -> Path:
    """
    Save length-controlled dataset to disk.

    Args:
        df: controlled dataset.
        output_path: where to save (CSV format).

    Returns:
        Path to saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.info(f"Saved controlled dataset to {output_path}")

    return output_path


def load_and_control_dataset(input_path: str | Path,
                              output_path: str | Path = "data/processed/spans_length_controlled.csv",
                              text_column: str = "text",
                              target_tokens: int = 100) -> Tuple[pd.DataFrame, Dict]:
    """
    Load processed dataset, apply length control, save, and report stats.

    Complete pipeline: load → control → save → report.

    Args:
        input_path: path to processed dataset.
        output_path: where to save controlled dataset.
        text_column: name of text column.
        target_tokens: target token length.

    Returns:
        Tuple of (controlled_df, statistics_dict).
    """
    logger.info(f"Loading dataset from {input_path}")
    df_original = pd.read_csv(input_path)

    logger.info(f"Loaded {len(df_original)} samples")

    # Apply length control
    df_controlled = apply_length_control(
        df_original,
        text_column=text_column,
        target_tokens=target_tokens
    )

    # Report statistics
    stats = report_length_statistics(df_original, df_controlled)

    # Log statistics
    logger.info("=" * 60)
    logger.info("LENGTH CONTROL STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Samples: {stats['n_samples']}")
    logger.info(f"Papers: {stats['n_papers']}")
    logger.info(f"Original tokens - mean: {stats['original_token_stats']['mean']:.1f}, "
                f"std: {stats['original_token_stats']['std']:.1f}")
    logger.info(f"Controlled tokens - mean: {stats['controlled_token_stats']['mean']:.1f}, "
                f"std: {stats['controlled_token_stats']['std']:.1f}")
    logger.info(f"Class distribution (controlled): {stats['class_counts']['controlled']}")
    logger.info("=" * 60)

    # Save
    save_controlled_dataset(df_controlled, output_path)

    return df_controlled, stats
