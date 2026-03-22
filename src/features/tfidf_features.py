"""TF-IDF feature extraction."""

import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def compute_tfidf_features(texts: list[str],
                            max_features: int = 5000,
                            ngram_range: tuple = (1, 2)) -> tuple[np.ndarray, TfidfVectorizer]:
    """
    Compute TF-IDF features for texts.

    Args:
        texts: list of text strings.
        max_features: maximum vocabulary size.
        ngram_range: (min_n, max_n) for n-grams.

    Returns:
        Tuple of (feature matrix, fitted vectorizer).
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=1,
        max_df=0.95,
        strip_accents="unicode",
        lowercase=True,
        analyzer="word",
        token_pattern=r"\b\w+\b"
    )

    X = vectorizer.fit_transform(texts).toarray()
    logger.info(f"TF-IDF: {len(texts)} texts, {X.shape[1]} features")

    return X, vectorizer


def extract_tfidf_dataset(df: pd.DataFrame,
                          text_column: str = "text",
                          max_features: int = 5000) -> tuple[np.ndarray, TfidfVectorizer]:
    """
    Extract TF-IDF features from dataset.

    Args:
        df: dataset with text column.
        text_column: name of text column.
        max_features: vocabulary size limit.

    Returns:
        Tuple of (feature matrix, fitted vectorizer).
    """
    texts = df[text_column].astype(str).tolist()
    return compute_tfidf_features(texts, max_features=max_features)
