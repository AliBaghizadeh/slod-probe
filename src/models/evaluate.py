"""Evaluation metrics and result saving."""

import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

logger = logging.getLogger(__name__)

LABEL_NAMES = ["macro", "meso", "micro"]


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    labels: Optional[list[str]] = None) -> Dict[str, float]:
    """
    Compute classification metrics.

    Metrics:
    - accuracy
    - macro F1 (computed over all three classes: macro, meso, micro)
    - per-class precision, recall, F1

    Args:
        y_true: true labels.
        y_pred: predicted labels.
        labels: optional label names (default: macro, meso, micro).

    Returns:
        Dict with metric names -> values.
    """
    if labels is None:
        labels = LABEL_NAMES

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    label_indices = list(range(len(labels)))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    # Per-class metrics (using one-vs-rest approach)
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    for i, label_name in enumerate(labels):
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        support = int(y_true_binary.sum())

        precision = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
        recall = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
        f1 = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))

        metrics[f"class_{label_name}_precision"] = precision
        metrics[f"class_{label_name}_recall"] = recall
        metrics[f"class_{label_name}_f1"] = f1
        metrics[f"class_{label_name}_support"] = support

        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)

    metrics["macro_precision"] = float(np.mean(per_class_precision))
    metrics["macro_recall"] = float(np.mean(per_class_recall))
    metrics["macro_f1"] = float(np.mean(per_class_f1))

    # Log results
    logger.info(f"Metrics: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
    logger.info(f"Per-class metrics:")
    for label_name in labels:
        logger.info(
            f"  {label_name}: precision={metrics[f'class_{label_name}_precision']:.4f}, "
            f"recall={metrics[f'class_{label_name}_recall']:.4f}, "
            f"f1={metrics[f'class_{label_name}_f1']:.4f}"
        )

    return metrics


def save_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          output_path: str | Path,
                          labels: Optional[list[str]] = None) -> None:
    """
    Compute and save confusion matrix as image.

    Uses fixed label order [0, 1, 2] (macro, meso, micro) for consistency.

    Args:
        y_true: true labels.
        y_pred: predicted labels.
        output_path: path to save figure.
        labels: optional label names.
    """
    if labels is None:
        labels = LABEL_NAMES

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100)
    plt.close()

    logger.info(f"Confusion matrix saved to {output_path}")


def save_confusion_matrix_table(y_true: np.ndarray,
                                y_pred: np.ndarray,
                                output_path: str | Path,
                                labels: Optional[list[str]] = None) -> None:
    """
    Save the raw confusion matrix counts as a CSV table.

    Rows are true labels and columns are predicted labels.
    """
    if labels is None:
        labels = LABEL_NAMES

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    index = [f"true_{label}" for label in labels]
    columns = [f"pred_{label}" for label in labels]
    df = pd.DataFrame(cm, index=index, columns=columns)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)

    logger.info(f"Confusion matrix table saved to {output_path}")


def save_predictions(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     output_path: str | Path,
                     metadata: Optional[Dict] = None) -> None:
    """
    Save predictions to CSV.

    Args:
        y_true: true labels.
        y_pred: predicted labels.
        output_path: path to save CSV.
        metadata: optional metadata columns (dict of column_name -> array).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "y_true": y_true,
        "y_pred": y_pred,
        "correct": y_true == y_pred,
    }

    if metadata:
        data.update(metadata)

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

    logger.info(f"Predictions saved to {output_path}")


def save_metrics_json(metrics: Dict[str, float],
                      output_path: str | Path) -> None:
    """
    Save metrics to JSON.

    Args:
        metrics: metrics dictionary.
        output_path: path to save JSON.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved to {output_path}")
