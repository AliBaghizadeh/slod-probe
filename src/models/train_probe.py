"""Training utilities for linear probes on cached embeddings."""

import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from typing import Tuple, Optional, Dict, Any
from datetime import datetime

from src.data.weak_labels import encode_label_series

logger = logging.getLogger(__name__)


def prepare_train_test_split(X: np.ndarray,
                              y: np.ndarray,
                              test_size: float = 0.2,
                              random_state: int = 42,
                              paper_ids: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets.

    Supports paper-level split to avoid leakage when paper_ids are provided.

    Args:
        X: feature matrix (n_samples, n_features).
        y: labels (n_samples,).
        test_size: fraction for test set.
        random_state: random seed.
        paper_ids: optional paper IDs for paper-level split (n_samples,).

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    if paper_ids is not None:
        # Paper-level split
        unique_papers = np.unique(paper_ids)
        n_papers = len(unique_papers)
        n_test_papers = max(1, int(np.ceil(n_papers * test_size)))

        np.random.seed(random_state)
        test_papers = np.random.choice(unique_papers, size=n_test_papers, replace=False)
        test_mask = np.isin(paper_ids, test_papers)

        X_train = X[~test_mask]
        X_test = X[test_mask]
        y_train = y[~test_mask]
        y_test = y[test_mask]

        logger.info(f"Paper-level split: {n_papers} papers -> {len(np.unique(paper_ids[~test_mask]))} train, {len(test_papers)} test")
    else:
        # Simple random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(f"Random split: {len(y_train)} train, {len(y_test)} test")

    return X_train, X_test, y_train, y_test


def train_probe(X_train: np.ndarray,
                y_train: np.ndarray,
                probe_type: str = "logistic",
                **kwargs) -> object:
    """
    Train a linear probe.

    Args:
        X_train: training features (n_samples, n_features).
        y_train: training labels.
        probe_type: type of probe (logistic, ridge, svm).
        **kwargs: additional arguments for probe.

    Returns:
        fitted probe model.
    """
    if probe_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(
            max_iter=kwargs.get("max_iter", 1000),
            random_state=kwargs.get("random_state", 42),
            solver="lbfgs"
        )
    elif probe_type == "ridge":
        from sklearn.linear_model import RidgeClassifier
        model = RidgeClassifier(
            alpha=kwargs.get("alpha", 1.0),
            random_state=kwargs.get("random_state", 42)
        )
    elif probe_type == "svm":
        from sklearn.svm import LinearSVC
        model = LinearSVC(
            max_iter=kwargs.get("max_iter", 1000),
            random_state=kwargs.get("random_state", 42)
        )
    else:
        raise ValueError(f"Unknown probe_type: {probe_type}")

    model.fit(X_train, y_train)
    logger.info(f"Trained {probe_type} probe: train_acc={model.score(X_train, y_train):.4f}")

    return model


def load_cached_embeddings(embedding_path: str | Path,
                          metadata_path: Optional[str | Path] = None) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
    """
    Load cached embeddings and optional metadata.

    Args:
        embedding_path: path to embeddings (.npy file).
        metadata_path: optional path to metadata CSV.

    Returns:
        Tuple of (embeddings array, metadata dataframe or None).
    """
    # Load embeddings
    logger.info(f"Loading embeddings from {embedding_path}")
    embeddings = np.load(embedding_path)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Load metadata if provided
    metadata = None
    if metadata_path and Path(metadata_path).exists():
        logger.info(f"Loading metadata from {metadata_path}")
        metadata = pd.read_csv(metadata_path)
        logger.info(f"Metadata shape: {metadata.shape}")

    return embeddings, metadata


def run_embedding_probe(embedding_path: str | Path,
                       labels: np.ndarray,
                       paper_ids: Optional[np.ndarray] = None,
                       metadata_path: Optional[str | Path] = None,
                       output_dir: str | Path = "results/embedding_probe",
                       test_size: float = 0.2,
                       random_state: int = 42) -> Dict[str, Any]:
    """
    Run logistic regression probe on cached embeddings.

    Complete pipeline: load embeddings → split → train → predict → save results.

    Args:
        embedding_path: path to embeddings array.
        labels: label array (n_samples,) or label dataframe column name.
        paper_ids: optional paper IDs for paper-level split.
        metadata_path: optional path to metadata CSV.
        output_dir: directory to save results.
        test_size: fraction for test set.
        random_state: random seed.

    Returns:
        Dict with results: model, predictions, metrics, metadata.
    """
    from src.models.evaluate import compute_metrics, save_confusion_matrix, save_predictions, save_metrics_json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Running Logistic Regression Probe on Cached Embeddings")
    logger.info("=" * 60)

    # Load embeddings and metadata
    embeddings, metadata = load_cached_embeddings(embedding_path, metadata_path)

    # Ensure labels is numpy array
    if isinstance(labels, str) and metadata is not None:
        logger.info(f"Converting label column '{labels}' from metadata")
        if labels in metadata.columns:
            labels_array = encode_label_series(metadata[labels]).to_numpy()
            logger.info("Label mapping: {0: 'macro', 1: 'meso', 2: 'micro'}")
        else:
            raise ValueError(f"Label column '{labels}' not found in metadata")
    else:
        labels_array = labels

    # Paper-level split
    logger.info(f"Split config: test_size={test_size}, random_state={random_state}")
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        embeddings,
        labels_array,
        test_size=test_size,
        random_state=random_state,
        paper_ids=paper_ids
    )

    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")

    # Train logistic regression
    logger.info("Training logistic regression probe...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
        verbose=0
    )
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    logger.info(f"Train accuracy: {train_acc:.4f}")
    logger.info(f"Test accuracy: {test_acc:.4f}")

    # Predict on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_metrics(y_test, y_pred)
    logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")

    # Save results
    logger.info(f"Saving results to {output_dir}")

    # Save metrics
    save_metrics_json(metrics, output_dir / "metrics.json")

    # Save predictions
    pred_data = {
        "y_true": y_test,
        "y_pred": y_pred,
    }
    save_predictions(y_test, y_pred, output_dir / "predictions.csv", metadata=pred_data)

    # Save confusion matrix
    save_confusion_matrix(y_test, y_pred, output_dir / "confusion_matrix.png")

    # Save run summary
    run_summary = {
        "timestamp": datetime.now().isoformat(),
        "model": "logistic_regression",
        "embedding_path": str(embedding_path),
        "n_samples": len(embeddings),
        "embedding_dim": embeddings.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_size": test_size,
        "random_state": random_state,
        "paper_level_split": paper_ids is not None,
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "metrics": metrics,
    }
    with open(output_dir / "run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)
    logger.info(f"Saved run summary to {output_dir / 'run_summary.json'}")

    logger.info("=" * 60)
    logger.info("Probe training complete!")
    logger.info("=" * 60)

    return {
        "model": model,
        "y_pred": y_pred,
        "y_test": y_test,
        "metrics": metrics,
        "metadata": metadata,
    }
