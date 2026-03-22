"""Preprocessing and cleaning of extracted spans."""

import logging
import re
import unicodedata
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def clean_text(text: str, remove_section_title: bool = False) -> str:
    """
    Clean and normalize text.

    - remove extra whitespace
    - normalize unicode (NFKD)
    - optionally remove section titles

    Args:
        text: raw text.
        remove_section_title: if True, remove first line if it looks like a title.

    Returns:
        cleaned text.
    """
    if not isinstance(text, str):
        return ""

    # Normalize unicode (NFKD decomposes accented characters)
    text = unicodedata.normalize("NFKD", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    # Optionally remove section title (first line if it's short and capitalized)
    if remove_section_title and text:
        lines = text.split("\n")
        if lines:
            first_line = lines[0].strip()
            # Heuristic: section titles are typically < 100 chars and mostly uppercase
            if len(first_line) < 100 and sum(1 for c in first_line if c.isupper()) > len(first_line) * 0.5:
                text = " ".join(lines[1:]).strip()

    return text


def count_tokens(text: str, tokenizer: str = "simple") -> int:
    """
    Count tokens in text.

    Simple tokenizer: split on whitespace.
    Other tokenizers (bert, etc.) not implemented in phase 1.

    Args:
        text: input text.
        tokenizer: tokenization method (simple, bert, etc.).

    Returns:
        token count.
    """
    if not isinstance(text, str) or not text:
        return 0

    if tokenizer == "simple":
        return len(text.split())

    # For phase 1, only simple tokenizer supported
    logger.warning(f"Tokenizer '{tokenizer}' not implemented, using simple")
    return len(text.split())


def preprocess_dataset(df: pd.DataFrame,
                      text_column: str = "text",
                      section_column: Optional[str] = None,
                      remove_section_title: bool = False) -> pd.DataFrame:
    """
    Preprocess entire dataset.

    - clean text
    - compute token counts
    - remove rows with empty text after cleaning
    - log dropped rows

    Args:
        df: raw dataset.
        text_column: name of text column.
        section_column: optional name of section column (currently unused).
        remove_section_title: whether to remove titles from text.

    Returns:
        cleaned dataset with added token_count column.
    """
    df = df.copy()
    initial_count = len(df)

    # Clean text
    df[text_column] = df[text_column].apply(
        lambda x: clean_text(x, remove_section_title=remove_section_title)
    )

    # Compute token counts
    df["token_count"] = df[text_column].apply(count_tokens)

    # Remove empty text rows
    non_empty = df[df[text_column].str.len() > 0]
    dropped = initial_count - len(non_empty)

    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with empty text after cleaning")

    return non_empty
