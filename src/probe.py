"""Prototype-compatible all-in-one probe runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import export_prototype_spans
from src.features.embed_text import embed_dataset
from src.features.tfidf_features import extract_tfidf_dataset
from src.models.evaluate import (
    compute_metrics,
    save_confusion_matrix,
    save_confusion_matrix_table,
    save_metrics_json,
    save_predictions,
)
from src.models.train_probe import prepare_train_test_split, train_probe
from src.utils.seed import set_seed


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_condition_paths(config: dict[str, Any], condition: str) -> tuple[Path, Path]:
    path_key = "in_domain_spans" if condition == "in_domain" else "controlled_spans"
    export_name = "refined_qasper_slod_in_domain.csv" if condition == "in_domain" else "refined_qasper_slod_controlled.csv"

    dataset_path = Path(config["paths"][path_key])
    export_path = Path(config["paths"]["prototype_spans_dir"]) / export_name
    return dataset_path, export_path


def resolve_output_dir(config: dict[str, Any], condition: str, model_key: str) -> Path:
    return Path(config["paths"]["prototype_results_base"]) / condition / model_key


def run_embedding_probe(
    df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    cache_dir: Path,
    batch_size: int,
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    embeddings, _ = embed_dataset(
        data_path=df,
        text_column="text",
        model_name=model_name,
        batch_size=batch_size,
        cache_dir=cache_dir,
        metadata_columns=["span_id", "paper_id", "weak_label", "section_name"],
        resume=True,
    )
    y = df["weak_label"].map({"macro": 0, "meso": 1, "micro": 2}).to_numpy()
    paper_ids = df["paper_id"].to_numpy()

    X_train, X_test, y_train, y_test = prepare_train_test_split(
        embeddings,
        y,
        test_size=test_size,
        random_state=random_state,
        paper_ids=paper_ids,
    )
    model = train_probe(X_train, y_train, probe_type="logistic", max_iter=1000, random_state=random_state)
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_metrics_json(metrics, output_dir / "metrics.json")
    save_predictions(y_test, y_pred, output_dir / "predictions.csv")
    save_confusion_matrix(y_test, y_pred, output_dir / "confusion_matrix.png")
    save_confusion_matrix_table(y_test, y_pred, output_dir / "confusion_matrix.csv")
    return metrics


def run_tfidf_probe(
    df: pd.DataFrame,
    output_dir: Path,
    max_features: int,
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    X, _ = extract_tfidf_dataset(df, text_column="text", max_features=max_features)
    y = df["weak_label"].map({"macro": 0, "meso": 1, "micro": 2}).to_numpy()
    paper_ids = df["paper_id"].to_numpy()

    X_train, X_test, y_train, y_test = prepare_train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        paper_ids=paper_ids,
    )
    model = train_probe(X_train, y_train, probe_type="logistic", max_iter=1000, random_state=random_state)
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_metrics_json(metrics, output_dir / "metrics.json")
    save_predictions(y_test, y_pred, output_dir / "predictions.csv")
    save_confusion_matrix(y_test, y_pred, output_dir / "confusion_matrix.png")
    save_confusion_matrix_table(y_test, y_pred, output_dir / "confusion_matrix.csv")
    return metrics


def write_condition_summary(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
    else:
        summary = {"condition": payload["condition"], "models": {}}

    summary["models"][payload["model"]] = {
        "dataset": payload["dataset"],
        "prototype_spans": payload["prototype_spans"],
        "results_dir": payload["results_dir"],
        "metrics": payload["metrics"],
    }

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prototype-compatible SLoD probe runner")
    parser.add_argument("--config", default="configs/prototype_scibert.yaml", help="Prototype YAML config")
    parser.add_argument("--train", action="store_true", help="Train the probe")
    parser.add_argument("--eval", action="store_true", help="Evaluate and save results")
    parser.add_argument("--condition", choices=["in_domain", "controlled"], default="in_domain")
    parser.add_argument("--model", choices=["scibert", "minilm", "tfidf"], default="scibert")
    args = parser.parse_args()

    if not args.train and not args.eval:
        raise SystemExit("Use at least one of --train or --eval")

    config = load_config(args.config)
    set_seed(config["split"]["random_state"])

    dataset_path, export_path = resolve_condition_paths(config, args.condition)
    df = pd.read_csv(dataset_path)
    export_prototype_spans(dataset_path, export_path)

    output_dir = resolve_output_dir(config, args.condition, args.model)
    metrics: dict[str, Any]

    if args.model == "tfidf":
        metrics = run_tfidf_probe(
            df=df,
            output_dir=output_dir,
            max_features=config.get("tfidf", {}).get("max_features", 5000),
            test_size=config["split"]["test_size"],
            random_state=config["split"]["random_state"],
        )
    else:
        model_config = config["models"][args.model]
        cache_key = "cache_in_domain" if args.condition == "in_domain" else "cache_controlled"
        metrics = run_embedding_probe(
            df=df,
            output_dir=output_dir,
            model_name=model_config["model_name"],
            cache_dir=Path(model_config[cache_key]),
            batch_size=model_config.get("batch_size", 16),
            test_size=config["split"]["test_size"],
            random_state=config["split"]["random_state"],
        )

    condition_dir = Path(config["paths"]["prototype_results_base"]) / args.condition
    write_condition_summary(
        condition_dir,
        {
            "condition": args.condition,
            "model": args.model,
            "dataset": str(dataset_path).replace("\\", "/"),
            "prototype_spans": str(export_path).replace("\\", "/"),
            "results_dir": str(output_dir).replace("\\", "/"),
            "metrics": metrics,
        },
    )

    print(f"Condition: {args.condition}")
    print(f"Model: {args.model}")
    print(f"Prototype spans: {export_path}")
    print(f"Results: {output_dir}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Meso F1: {metrics['class_meso_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
