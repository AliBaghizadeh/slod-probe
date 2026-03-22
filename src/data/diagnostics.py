"""Dataset diagnostics and readiness checks."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

REQUIRED_COLUMNS = ["paper_id", "domain", "section_name", "span_id", "text", "weak_label", "token_count"]


def load_spans_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a processed spans CSV and validate required columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    return df


def _numeric_summary(series: pd.Series) -> Dict[str, float]:
    return {
        "min": int(series.min()),
        "max": int(series.max()),
        "mean": float(series.mean()),
        "median": float(series.median()),
    }


def compute_diagnostics(df: pd.DataFrame) -> Dict:
    """Compute summary diagnostics for a weak-labeled span dataset."""
    class_counts = df["weak_label"].value_counts(dropna=False).to_dict()
    spans_per_paper = df.groupby("paper_id").size()
    token_counts_by_class = {
        label: {
            **_numeric_summary(group["token_count"]),
            "count": int(len(group)),
        }
        for label, group in df.groupby("weak_label")
    }

    duplicate_counts = df["text"].value_counts()
    duplicate_examples = [
        {
            "count": int(count),
            "preview": text[:120],
        }
        for text, count in duplicate_counts[duplicate_counts > 1].head(10).items()
    ]

    section_distribution_by_class = {
        label: group["section_name"].value_counts().to_dict()
        for label, group in df.groupby("weak_label")
    }

    confidence_distribution = {}
    confidence_distribution_by_class = {}
    if "weak_label_confidence" in df.columns:
        confidence_distribution = {
            str(key): int(value)
            for key, value in df["weak_label_confidence"].value_counts(dropna=False).to_dict().items()
        }
        confidence_distribution_by_class = {
            label: {
                str(key): int(value)
                for key, value in group["weak_label_confidence"].value_counts(dropna=False).to_dict().items()
            }
            for label, group in df.groupby("weak_label")
        }

    diagnostics = {
        "total_papers": int(df["paper_id"].nunique()),
        "total_spans": int(len(df)),
        "class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "spans_per_paper": {
            **_numeric_summary(spans_per_paper),
            "papers_below_5_spans": int((spans_per_paper < 5).sum()),
        },
        "token_count_by_class": token_counts_by_class,
        "duplicate_texts": {
            "num_duplicate_texts": int((duplicate_counts > 1).sum()),
            "duplicate_rows": int((duplicate_counts[duplicate_counts > 1] - 1).sum()),
            "examples": duplicate_examples,
        },
        "section_distribution_by_class": section_distribution_by_class,
    }

    if confidence_distribution:
        diagnostics["label_confidence_distribution"] = confidence_distribution
        diagnostics["label_confidence_by_class"] = confidence_distribution_by_class

    return diagnostics


def evaluate_dataset_readiness(
    diagnostics: Dict,
    min_papers: int = 100,
    min_total_spans: int = 240,
    min_spans_per_class: int = 80,
    min_tokens_per_span: int = 60,
    max_tokens_per_span: int = 180,
) -> Dict:
    """Turn diagnostics into explicit errors/warnings for experiment gating."""
    class_counts = diagnostics.get("class_counts", {})
    token_count_by_class = diagnostics.get("token_count_by_class", {})

    errors = []
    warnings = []

    if diagnostics["total_papers"] < min_papers:
        errors.append(f"Only {diagnostics['total_papers']} papers available; need at least {min_papers}.")

    if diagnostics["total_spans"] < min_total_spans:
        errors.append(f"Only {diagnostics['total_spans']} spans available; need at least {min_total_spans}.")

    for label in ("macro", "meso", "micro"):
        count = int(class_counts.get(label, 0))
        if count < min_spans_per_class:
            errors.append(f"Class '{label}' has {count} spans; need at least {min_spans_per_class}.")

    for label, summary in token_count_by_class.items():
        if summary["min"] < min_tokens_per_span:
            warnings.append(
                f"Class '{label}' contains spans shorter than {min_tokens_per_span} tokens (min={summary['min']})."
            )
        if summary["max"] > max_tokens_per_span:
            warnings.append(
                f"Class '{label}' contains spans longer than {max_tokens_per_span} tokens (max={summary['max']})."
            )

    nonzero_class_counts = [int(class_counts.get(label, 0)) for label in ("macro", "meso", "micro") if int(class_counts.get(label, 0)) > 0]
    if len(nonzero_class_counts) >= 2:
        imbalance_ratio = max(nonzero_class_counts) / min(nonzero_class_counts)
        if imbalance_ratio > 2.5:
            warnings.append(f"Class imbalance is {imbalance_ratio:.2f}x between largest and smallest class.")

    duplicate_info = diagnostics.get("duplicate_texts", {})
    if duplicate_info.get("duplicate_rows", 0) > 0:
        warnings.append(
            f"Found {duplicate_info['duplicate_rows']} duplicate rows across {duplicate_info['num_duplicate_texts']} duplicate texts."
        )

    readiness = {
        "ready": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }
    return readiness


def save_diagnostics_json(payload: Dict, output_path: str | Path) -> Path:
    """Persist diagnostics or readiness payloads to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


def format_diagnostics_report(diagnostics: Dict, readiness: Dict | None = None) -> str:
    """Create a compact human-readable diagnostics report."""
    class_counts = diagnostics["class_counts"]
    spans_per_paper = diagnostics["spans_per_paper"]
    duplicate_info = diagnostics["duplicate_texts"]

    lines = [
        "=" * 80,
        "DATASET DIAGNOSTICS",
        "=" * 80,
        f"Total papers: {diagnostics['total_papers']}",
        f"Total spans:  {diagnostics['total_spans']}",
        "",
        "Class counts:",
    ]

    for label in ("macro", "meso", "micro"):
        lines.append(f"  {label:5s}: {int(class_counts.get(label, 0))}")

    lines.extend(
        [
            "",
            "Spans per paper:",
            f"  min={spans_per_paper['min']} max={spans_per_paper['max']} mean={spans_per_paper['mean']:.1f}",
            "",
            "Token counts by class:",
        ]
    )

    for label in ("macro", "meso", "micro"):
        summary = diagnostics["token_count_by_class"].get(label)
        if summary:
            lines.append(
                f"  {label:5s}: count={summary['count']} min={summary['min']} max={summary['max']} mean={summary['mean']:.1f}"
            )

    lines.extend(
        [
            "",
            f"Duplicate texts: {duplicate_info['num_duplicate_texts']} ({duplicate_info['duplicate_rows']} duplicate rows)",
        ]
    )

    if "label_confidence_distribution" in diagnostics:
        lines.extend(["", "Label confidence distribution:"])
        for confidence, count in diagnostics["label_confidence_distribution"].items():
            lines.append(f"  {confidence:5s}: {count}")

        lines.extend(["", "Confidence by class:"])
        for label in ("macro", "meso", "micro"):
            confidence_counts = diagnostics["label_confidence_by_class"].get(label, {})
            rendered = ", ".join(f"{name}={count}" for name, count in sorted(confidence_counts.items()))
            lines.append(f"  {label:5s}: {rendered or 'none'}")

    lines.extend(["", "Top sections by class:"])

    for label in ("macro", "meso", "micro"):
        section_counts = diagnostics["section_distribution_by_class"].get(label, {})
        top_sections = sorted(section_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
        rendered = ", ".join(f"{name}={count}" for name, count in top_sections) if top_sections else "none"
        lines.append(f"  {label:5s}: {rendered}")

    if readiness is not None:
        lines.extend(["", f"Ready: {'yes' if readiness['ready'] else 'no'}"])
        if readiness["errors"]:
            lines.append("Errors:")
            lines.extend(f"  - {error}" for error in readiness["errors"])
        if readiness["warnings"]:
            lines.append("Warnings:")
            lines.extend(f"  - {warning}" for warning in readiness["warnings"])

    lines.append("=" * 80)
    return "\n".join(lines)
