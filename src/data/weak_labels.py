"""Weak-label mapping and validation."""

import logging
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

LABEL_TO_ID = {"macro": 0, "meso": 1, "micro": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def normalize_section_name(section_name: str) -> str:
    """
    Normalize section name for mapping.

    - lowercase
    - strip whitespace
    - handle common aliases

    Args:
        section_name: raw section name.

    Returns:
        normalized section name.
    """
    if not section_name or not isinstance(section_name, str):
        return ""

    norm = section_name.strip().lower()

    # Handle common aliases
    aliases = {
        "abstract": "abstract",
        "summary": "abstract",
        "intro": "introduction",
        "introduction": "introduction",
        "related work": "related work",
        "literature": "related work",
        "method": "methods",
        "methods": "methods",
        "methodology": "methods",
        "approach": "methods",
        "algorithm": "methods",
        "result": "results",
        "results": "results",
        "experiments": "results",
        "evaluation": "results",
        "experiment": "results",
        "discussion": "discussion",
        "conclusion": "conclusion",
        "conclusions": "conclusion",
        "future work": "conclusion",
    }

    return aliases.get(norm, norm)


def get_label_mapping(custom_mapping: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Return mapping from section names to weak labels.

    Improved strategy for conservative meso coverage:
    - macro: abstract, introduction, conclusion, summary
    - micro: methods, results, experiments, implementation, approach, algorithm
    - meso: related work, discussion, background, motivation, high-level framing sections

    Args:
        custom_mapping: optional custom section -> label mapping.

    Returns:
        Dict mapping normalized section names to (macro, meso, micro).
    """
    default_mapping = {
        # Macro: High-level overview and conclusion
        "abstract": "macro",
        "introduction": "macro",
        "conclusion": "macro",
        "summary": "macro",

        # Micro: Detailed technical content
        "methods": "micro",
        "methodology": "micro",
        "results": "micro",
        "experiments": "micro",
        "experimental setup": "micro",
        "implementation": "micro",
        "implementation details": "micro",
        "approach": "micro",
        "algorithm": "micro",
        "algorithms": "micro",
        "technical details": "micro",
        "training details": "micro",
        "computational analysis": "micro",
        "ablations": "micro",
        "ablation": "micro",

        # Meso: Organizational and contextual content
        "related work": "meso",
        "literature": "meso",
        "discussion": "meso",
        "analysis": "meso",
        "background": "meso",
        "motivation": "meso",
        "annotation guidelines": "meso",
        "problem formulation": "meso",
        "method overview": "meso",
        "convergence analysis": "meso",
        "theoretical framework": "meso",
        "proof sketch": "meso",
    }

    if custom_mapping:
        default_mapping.update(custom_mapping)

    return default_mapping


def section_to_label(section_name: str,
                     label_mapping: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    Map section name to weak label.

    Returns None for unknown sections.

    Args:
        section_name: raw section name.
        label_mapping: optional custom mapping (uses default if None).

    Returns:
        Label (macro, meso, micro) or None if unknown.
    """
    if label_mapping is None:
        label_mapping = get_label_mapping()

    norm = normalize_section_name(section_name)
    return label_mapping.get(norm, None)


def validate_labels(label_series: pd.Series) -> bool:
    """
    Check that labels are valid and log counts.

    Logs label distribution.

    Args:
        label_series: pandas Series with labels.

    Returns:
        True if all labels are in {macro, meso, micro, None}, False otherwise.
    """
    valid_labels = {"macro", "meso", "micro", None}
    invalid = label_series[~label_series.isin(valid_labels)]

    if len(invalid) > 0:
        logger.error(f"Found {len(invalid)} invalid labels: {invalid.unique()}")
        return False

    logger.info("Label validation passed.")
    return True


def get_label_distribution(label_series: pd.Series) -> Dict[str, int]:
    """
    Get counts for each label.

    Includes None for unmapped sections.

    Args:
        label_series: pandas Series with labels.

    Returns:
        Dict with label -> count.
    """
    distribution = {}
    for label, count in label_series.value_counts(dropna=False).items():
        normalized_label = None if pd.isna(label) else label
        distribution[normalized_label] = int(count)
    return distribution


def encode_label_series(label_series: pd.Series) -> pd.Series:
    """
    Map semantic labels to a stable numeric id.

    Returns:
        Series with macro=0, meso=1, micro=2.
    """
    unknown = sorted(
        label for label in label_series.dropna().unique()
        if label not in LABEL_TO_ID
    )
    if unknown:
        raise ValueError(f"Unknown labels encountered: {unknown}")

    encoded = label_series.map(LABEL_TO_ID)
    if encoded.isna().any():
        raise ValueError("Encountered missing labels while encoding")

    return encoded.astype(int)
