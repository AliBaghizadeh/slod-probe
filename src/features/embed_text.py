"""Text embedding extraction and caching using sentence-transformers."""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Default model
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32


def check_gpu_available() -> bool:
    """
    Check if GPU is available for torch.

    Returns:
        True if GPU is available, False otherwise.
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def embed_texts(texts: list[str],
                model_name: str = DEFAULT_MODEL,
                batch_size: int = DEFAULT_BATCH_SIZE,
                show_progress: bool = False) -> np.ndarray:
    """
    Compute embeddings for texts using sentence-transformers.

    Loads model to GPU if available, otherwise CPU.

    Args:
        texts: list of text strings.
        model_name: huggingface model identifier (default: all-MiniLM-L6-v2).
        batch_size: batch size for encoding.
        show_progress: whether to show progress bar.

    Returns:
        Embedding matrix (n_samples, embedding_dim).
    """
    from sentence_transformers import SentenceTransformer

    # Detect device
    device = "cuda" if check_gpu_available() else "cpu"
    logger.info(f"Loading model {model_name} on {device.upper()}")

    # Load model
    model = SentenceTransformer(model_name)
    model = model.to(device)

    # Encode texts
    logger.info(f"Encoding {len(texts)} texts with batch_size={batch_size}")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        device=device,
        convert_to_numpy=True
    )

    logger.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def embed_dataset(data_path: str | Path | pd.DataFrame,
                  text_column: str = "text",
                  model_name: str = DEFAULT_MODEL,
                  batch_size: int = DEFAULT_BATCH_SIZE,
                  cache_dir: str | Path = "data/cache",
                  metadata_columns: str | list[str] | None = None,
                  resume: bool = True) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
    """
    Extract embeddings for entire dataset with caching.

    Supports both CSV file paths and pandas DataFrames.
    Saves embeddings to cache directory with aligned metadata.
    Resume-safe: skips extraction if cache files already exist.

    Args:
        data_path: path to CSV file OR pandas DataFrame.
        text_column: name of text column.
        model_name: embedding model (default: all-MiniLM-L6-v2).
        batch_size: batch size for encoding.
        cache_dir: directory to save embeddings and metadata.
        metadata_columns: column name (str) or list of column names to save as metadata.
        resume: if True, load from cache if it exists.

    Returns:
        Tuple of (embeddings_array, metadata_dataframe or None).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Generate cache file names
    cache_embeddings = cache_dir / f"embeddings_{model_name.replace('/', '_')}.npy"
    cache_metadata = cache_dir / f"metadata_{model_name.replace('/', '_')}.csv"
    cache_log = cache_dir / f"embedding_log_{model_name.replace('/', '_')}.json"

    # Load data if it's a path
    if isinstance(data_path, (str, Path)):
        logger.info(f"Loading dataset from {data_path}")
        df = pd.read_csv(data_path)
    else:
        df = data_path.copy()

    # Check resume
    if resume and cache_embeddings.exists() and cache_metadata.exists():
        logger.info(f"Loading cached embeddings from {cache_embeddings}")
        embeddings = load_embeddings(cache_embeddings)
        metadata = pd.read_csv(cache_metadata)
        logger.info(f"Loaded embeddings: {embeddings.shape}, metadata: {metadata.shape}")
        return embeddings, metadata

    # Extract embeddings
    texts = df[text_column].astype(str).tolist()
    embeddings = embed_texts(texts, model_name=model_name, batch_size=batch_size)

    # Prepare metadata if requested
    metadata = None
    if metadata_columns:
        # Handle both string and list inputs
        if isinstance(metadata_columns, str):
            cols = [metadata_columns]
        else:
            cols = metadata_columns

        missing = [c for c in cols if c not in df.columns]
        if missing:
            logger.warning(f"Metadata columns not found in dataset: {missing}")
            cols = [c for c in cols if c in df.columns]

        if cols:
            metadata = df[cols].copy()

    # Save embeddings
    logger.info(f"Saving embeddings to {cache_embeddings}")
    np.save(cache_embeddings, embeddings)

    # Save metadata if available
    if metadata is not None:
        logger.info(f"Saving metadata to {cache_metadata}")
        metadata.to_csv(cache_metadata, index=False)

    # Save log
    log_info = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "n_samples": len(texts),
        "embedding_dim": embeddings.shape[1],
        "batch_size": batch_size,
        "text_column": text_column,
        "metadata_columns": metadata_columns,
        "cache_embeddings": str(cache_embeddings),
        "cache_metadata": str(cache_metadata),
    }
    with open(cache_log, "w") as f:
        json.dump(log_info, f, indent=2)
    logger.info(f"Saved log to {cache_log}")

    return embeddings, metadata


def load_embeddings(cache_path: str | Path) -> np.ndarray:
    """
    Load cached embeddings from disk.

    Supports .npy and .npz formats.

    Args:
        cache_path: path to saved embeddings file.

    Returns:
        Embedding matrix (n_samples, embedding_dim).
    """
    cache_path = Path(cache_path)

    if not cache_path.exists():
        raise FileNotFoundError(f"Embeddings not found at {cache_path}")

    if cache_path.suffix == ".npy":
        embeddings = np.load(cache_path)
    elif cache_path.suffix == ".npz":
        data = np.load(cache_path)
        # Assume first array is embeddings
        embeddings = data["embeddings"] if "embeddings" in data else data[list(data.files)[0]]
    else:
        raise ValueError(f"Unsupported format: {cache_path.suffix}")

    logger.info(f"Loaded embeddings from {cache_path}: {embeddings.shape}")
    return embeddings
