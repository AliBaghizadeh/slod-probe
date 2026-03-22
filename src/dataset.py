"""Prototype-compatible dataset wrapper and export helpers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.diagnostics import compute_diagnostics, save_diagnostics_json
from src.data.qasper import save_qasper_spans

PROTOTYPE_REQUIRED_COLUMNS = [
    "paper_id",
    "domain",
    "section_name",
    "span_id",
    "text",
    "token_count",
    "weak_label",
]


def load_spans(path: str | Path) -> pd.DataFrame:
    """Load a processed spans CSV and validate the minimum schema."""
    df = pd.read_csv(path)
    missing = [column for column in PROTOTYPE_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Spans file missing required columns: {missing}")
    return df


def export_prototype_spans(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Export a prototype-formatted spans CSV under data/spans/."""
    df = load_spans(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_df = df[
        ["paper_id", "domain", "section_name", "span_id", "text", "token_count", "weak_label"]
    ].rename(columns={"weak_label": "label"})
    export_df.to_csv(output_path, index=False)
    return export_df


def load_yaml(path: str | Path | None) -> dict[str, Any]:
    """Load a YAML config if provided."""
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def extract_qasper_from_config(config_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run a QASPER extraction using a YAML config."""
    config = load_yaml(config_path)
    extraction = config.get("qasper_extraction", {})
    validation = config.get("dataset_validation", {})
    output_path = config.get("paths", {}).get("processed_spans", "data/spans/qasper_prototype_spans.csv")

    df, stats = save_qasper_spans(
        output_path=output_path,
        split=extraction.get("split", "train"),
        max_papers=extraction.get("max_papers", 100),
        min_tokens=extraction.get("min_tokens", 1),
        max_tokens=extraction.get("max_tokens", 150),
        cache_dir=extraction.get("cache_dir", "data/raw/qasper"),
        domain=extraction.get("domain", "scientific"),
        label_rules=extraction.get("label_rules"),
    )

    diagnostics = compute_diagnostics(df)
    diagnostics_output = validation.get(
        "diagnostics_output",
        str(Path(output_path).with_name(f"{Path(output_path).stem}_diagnostics.json")),
    )
    save_diagnostics_json(
        {
            "dataset_path": str(output_path),
            "extraction_stats": stats,
            "diagnostics": diagnostics,
        },
        diagnostics_output,
    )
    return df, {"stats": stats, "diagnostics_output": diagnostics_output}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prototype-compatible dataset helpers")
    parser.add_argument("--input", help="Input processed spans CSV")
    parser.add_argument("--output", help="Output prototype-formatted spans CSV")
    parser.add_argument("--export-prototype-spans", action="store_true", help="Export a prototype-formatted spans CSV")
    parser.add_argument("--extract-qasper", action="store_true", help="Run a QASPER extraction from config")
    parser.add_argument("--config", help="YAML config path")
    args = parser.parse_args()

    if args.export_prototype_spans:
        if not args.input or not args.output:
            raise SystemExit("--export-prototype-spans requires --input and --output")
        df = export_prototype_spans(args.input, args.output)
        print(f"Exported {len(df)} prototype spans to {args.output}")
        return 0

    if args.extract_qasper:
        if not args.config:
            raise SystemExit("--extract-qasper requires --config")
        df, info = extract_qasper_from_config(args.config)
        print(f"Extracted {len(df)} spans using prototype config {args.config}")
        print(f"Diagnostics saved to {info['diagnostics_output']}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
