"""Utilities for building a weak-labeled QASPER span dataset."""

from __future__ import annotations

import json
import math
import re
import tarfile
import urllib.request
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

QASPER_TRAIN_DEV_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
QASPER_TEST_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"

ARCHIVE_BY_SPLIT = {
    "train": ("qasper-train-dev-v0.3.tgz", QASPER_TRAIN_DEV_URL, "qasper-train-v0.3.json"),
    "dev": ("qasper-train-dev-v0.3.tgz", QASPER_TRAIN_DEV_URL, "qasper-dev-v0.3.json"),
    "test": ("qasper-test-and-evaluator-v0.3.tgz", QASPER_TEST_URL, "qasper-test-v0.3.json"),
}

OUTPUT_COLUMNS = [
    "paper_id",
    "domain",
    "section_name",
    "paragraph_index",
    "chunk_index",
    "span_id",
    "text",
    "weak_label",
    "weak_label_confidence",
    "weak_label_signal_count",
    "token_count",
]

DEFAULT_LABEL_RULES = {
    "mode": "refined",
    "include_title": False,
    "macro_intro_paragraph_limit": None,
    "prototype_lead_sentence_only": False,
    "prototype_micro_detail_section_keywords": [
        "methods",
        "method",
        "methodology",
        "experiments",
        "experiment",
        "results",
        "evaluation",
    ],
    "macro_section_keywords": [
        "abstract",
        "introduction",
        "conclusion",
        "conclusions",
        "concluding remarks",
        "discussion summary",
        "summary",
        "conclusion and future work",
        "conclusions and future work",
        "discussion and conclusion",
        "conclusion future work",
    ],
    "meso_section_keywords": [
        "results",
        "result",
        "analysis",
        "discussion",
        "approach",
        "model",
        "method overview",
        "overview",
        "background",
        "related work",
        "related works",
        "motivation",
        "problem formulation",
        "case study",
        "comparison",
        "comparisons",
        "main results",
        "performance analysis",
        "model analysis",
    ],
    "meso_excluded_section_keywords": [
        "implementation",
        "training",
        "hyperparameter",
        "experimental setup",
        "experimental settings",
        "experiment setup",
        "setup",
        "dataset",
        "datasets",
        "data collection",
        "data set",
        "evaluation metric",
        "evaluation protocol",
        "preprocessing",
        "statistics",
        "baseline",
        "baselines",
        "loss function",
        "optimizer",
        "architecture",
        "annotation",
        "corpus",
        "ablation",
        "parameter",
    ],
    "micro_section_keywords": [
        "methods",
        "method",
        "methodology",
        "experimental setup",
        "experimental settings",
        "implementation",
        "implementation details",
        "training",
        "training setup",
        "training details",
        "hyperparameters",
        "dataset",
        "datasets",
        "dataset details",
        "data collection",
        "preprocessing",
        "evaluation",
        "evaluation setup",
        "evaluation metrics",
        "architecture",
        "model architecture",
        "system architecture",
        "loss function",
        "baselines",
        "results",
        "analysis",
        "approach",
        "model",
        "experiments",
        "experiment",
    ],
    "ignore_section_keywords": [
        "acknowledg",
        "appendix",
        "reference",
        "supplement",
        "bibliography",
    ],
    "meso_excluded_text_patterns": [
        "learning rate",
        "batch size",
        "dropout",
        "optimizer",
        "adam",
        "sgd",
        "epochs",
        "epoch",
        "hidden size",
        "weight decay",
        "beam size",
        "gpu",
        "hyperparameter",
        "implementation details",
        "training setup",
        "training details",
        "evaluation setup",
        "dataset statistics",
        "train/dev/test",
        "training set",
        "validation set",
        "test set",
        "parameter setting",
        "parameters are",
    ],
    "meso_lexical_cues": [
        "in this section",
        "this section",
        "in this part",
        "we evaluate",
        "we analyze",
        "we compare",
        "we discuss",
        "we describe",
        "we present",
        "we investigate",
        "we examine",
        "we study",
        "we now describe",
        "we now analyze",
        "we now discuss",
        "we first",
        "we then",
        "we finally",
        "our approach consists of",
        "our method consists of",
        "we propose",
    ],
    "meso_high_confidence_min_signals": 3,
    "drop_low_confidence_meso": False,
    "drop_low_confidence_meso_only_if_micro_conflict": False,
    "meso_numeric_token_threshold": 8,
    "meso_numeric_token_ratio": 0.12,
}


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def tokenize_simple(text: str) -> List[str]:
    """Simple whitespace tokenization."""
    return normalize_whitespace(text).split()


def normalize_section_name(section_name: str) -> str:
    """Normalize section names for rule matching."""
    if not isinstance(section_name, str):
        return ""

    normalized = section_name.lower().replace("&", " and ").replace(":::", " ")
    normalized = re.sub(r"^\s*(?:\d+(?:\.\d+)*)[\s\-:.)]+", "", normalized)
    normalized = re.sub(r"^\s*(?:[ivxlcdm]+)[\s\-:.)]+", "", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def contains_keyword(section_name: str, keywords: Iterable[str]) -> bool:
    """Return True if any keyword appears in the normalized section name."""
    normalized = normalize_section_name(section_name)
    return any(keyword in normalized for keyword in keywords)


def find_matching_keywords(section_name: str, keywords: Iterable[str]) -> List[str]:
    """Return all matching normalized section keywords."""
    normalized = normalize_section_name(section_name)
    return [keyword for keyword in keywords if keyword in normalized]


def find_matching_phrases(text: str, phrases: Iterable[str]) -> List[str]:
    """Return lexical phrases that appear in normalized text."""
    normalized = normalize_whitespace(text).lower()
    if not normalized:
        return []
    return [phrase for phrase in phrases if phrase in normalized]


def resolve_label_rules(label_rules: Dict | None = None) -> Dict:
    """Merge custom label rules over the defaults."""
    rules = deepcopy(DEFAULT_LABEL_RULES)
    if label_rules:
        for key, value in label_rules.items():
            rules[key] = value
    return rules


def count_numeric_tokens(text: str) -> int:
    """Count tokens containing explicit numeric/configuration markers."""
    tokens = tokenize_simple(text)
    return sum(1 for token in tokens if re.search(r"\d", token))


def first_sentence(text: str) -> str:
    """Return the first sentence-like segment from text."""
    normalized = normalize_whitespace(text)
    if not normalized:
        return ""

    parts = re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)
    sentence = parts[0].strip()
    return sentence or normalized


def has_low_level_detail(text: str, label_rules: Dict) -> bool:
    """Detect configuration-heavy paragraphs that should not become meso."""
    normalized_text = normalize_whitespace(text).lower()
    if not normalized_text:
        return False

    if any(pattern in normalized_text for pattern in label_rules["meso_excluded_text_patterns"]):
        return True

    tokens = tokenize_simple(normalized_text)
    if not tokens:
        return False

    numeric_tokens = count_numeric_tokens(normalized_text)
    numeric_ratio = numeric_tokens / len(tokens)

    if numeric_tokens >= label_rules["meso_numeric_token_threshold"]:
        return True
    if numeric_ratio >= label_rules["meso_numeric_token_ratio"]:
        return True

    return False


def get_low_level_detail_signals(text: str, label_rules: Dict) -> List[str]:
    """Return detail cues that make a span too technical for meso."""
    normalized_text = normalize_whitespace(text).lower()
    signals = find_matching_phrases(normalized_text, label_rules["meso_excluded_text_patterns"])

    tokens = tokenize_simple(normalized_text)
    if tokens:
        numeric_tokens = count_numeric_tokens(normalized_text)
        numeric_ratio = numeric_tokens / len(tokens)
        if numeric_tokens >= label_rules["meso_numeric_token_threshold"]:
            signals.append("numeric_token_threshold")
        if numeric_ratio >= label_rules["meso_numeric_token_ratio"]:
            signals.append("numeric_token_ratio")

    return list(dict.fromkeys(signals))


def chunk_text_by_tokens(
    text: str,
    min_tokens: int = 60,
    max_tokens: int = 180,
) -> List[Tuple[str, int]]:
    """
    Convert a paragraph into one or more chunks within a token range.

    Paragraphs shorter than `min_tokens` are dropped. Paragraphs longer than
    `max_tokens` are split into near-even chunks so the trailing chunk is not tiny.
    """
    tokens = tokenize_simple(text)
    token_count = len(tokens)

    if token_count < min_tokens:
        return []

    if token_count <= max_tokens:
        return [(" ".join(tokens), token_count)]

    n_chunks = math.ceil(token_count / max_tokens)
    base_size = token_count // n_chunks
    remainder = token_count % n_chunks

    chunks: List[Tuple[str, int]] = []
    start = 0
    for chunk_idx in range(n_chunks):
        chunk_size = base_size + (1 if chunk_idx < remainder else 0)
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        if len(chunk_tokens) >= min_tokens:
            chunks.append((" ".join(chunk_tokens), len(chunk_tokens)))
        start = end

    return chunks


def assign_weak_label_metadata(
    section_name: str,
    paragraph_index: int,
    text: str = "",
    chunk_index: int = 0,
    label_rules: Dict | None = None,
) -> Dict[str, object]:
    """
    Apply hybrid weak-label rules and expose confidence metadata.

    Priority:
    1. Macro sections remain macro throughout.
    2. Lead paragraphs of designated discourse-heavy sections become meso
       only when they also show discourse-organizing lexical cues.
    3. Technical/detail sections become micro.
    4. Low-confidence meso candidates can be dropped entirely.
    """
    rules = resolve_label_rules(label_rules)
    normalized = normalize_section_name(section_name)
    is_lead_span = paragraph_index == 0 and chunk_index == 0

    result: Dict[str, object] = {
        "weak_label": None,
        "weak_label_confidence": None,
        "weak_label_signal_count": 0,
        "dropped_candidate_label": None,
        "drop_reason": None,
    }

    if not normalized:
        return result

    if contains_keyword(normalized, rules["ignore_section_keywords"]):
        result["drop_reason"] = "ignored_section"
        return result

    if rules.get("mode") == "prototype":
        intro_limit = rules.get("macro_intro_paragraph_limit")
        prototype_micro_sections = rules.get("prototype_micro_detail_section_keywords", [])

        if normalized == "title":
            result.update(
                {
                    "weak_label": "macro",
                    "weak_label_confidence": "high",
                    "weak_label_signal_count": 1,
                }
            )
            return result

        if contains_keyword(normalized, ["abstract", "conclusion", "conclusions"]):
            result.update(
                {
                    "weak_label": "macro",
                    "weak_label_confidence": "high",
                    "weak_label_signal_count": 1,
                }
            )
            return result

        if contains_keyword(normalized, ["introduction"]):
            if intro_limit is None or paragraph_index < int(intro_limit):
                result.update(
                    {
                        "weak_label": "macro",
                        "weak_label_confidence": "high",
                        "weak_label_signal_count": 1,
                    }
                )
            return result

        if is_lead_span:
            result.update(
                {
                    "weak_label": "meso",
                    "weak_label_confidence": "high",
                    "weak_label_signal_count": 1,
                }
            )
            return result

        if contains_keyword(normalized, prototype_micro_sections):
            result.update(
                {
                    "weak_label": "micro",
                    "weak_label_confidence": "high",
                    "weak_label_signal_count": 1,
                }
            )
            return result

        return result

    macro_matches = find_matching_keywords(normalized, rules["macro_section_keywords"])
    if macro_matches:
        result.update(
            {
                "weak_label": "macro",
                "weak_label_confidence": "high",
                "weak_label_signal_count": 1,
            }
        )
        return result

    meso_section_matches = find_matching_keywords(normalized, rules["meso_section_keywords"])
    micro_section_matches = find_matching_keywords(normalized, rules["micro_section_keywords"])
    meso_excluded_section_matches = find_matching_keywords(normalized, rules["meso_excluded_section_keywords"])
    meso_lexical_matches = find_matching_phrases(text, rules.get("meso_lexical_cues", []))
    detail_signals = get_low_level_detail_signals(text, rules)
    meso_excluded_by_text = bool(detail_signals)

    if is_lead_span and meso_section_matches and not meso_excluded_section_matches and not meso_excluded_by_text:
        meso_signal_count = int(bool(meso_section_matches)) + int(is_lead_span) + int(bool(meso_lexical_matches))
        confidence = "high" if meso_signal_count >= rules["meso_high_confidence_min_signals"] else "low"
        should_drop_low_confidence = confidence == "low" and rules.get("drop_low_confidence_meso", False)
        if should_drop_low_confidence and rules.get("drop_low_confidence_meso_only_if_micro_conflict", False):
            should_drop_low_confidence = bool(micro_section_matches)

        if should_drop_low_confidence:
            result.update(
                {
                    "dropped_candidate_label": "meso",
                    "drop_reason": "low_confidence_meso",
                }
            )
            return result

        result.update(
            {
                "weak_label": "meso",
                "weak_label_confidence": confidence,
                "weak_label_signal_count": meso_signal_count,
            }
        )
        return result

    if micro_section_matches or meso_excluded_by_text:
        high_confidence_micro = bool(detail_signals) or (bool(micro_section_matches) and not bool(meso_section_matches))
        signal_count = int(bool(micro_section_matches)) + int(bool(detail_signals))
        result.update(
            {
                "weak_label": "micro",
                "weak_label_confidence": "high" if high_confidence_micro else "low",
                "weak_label_signal_count": signal_count,
            }
        )
        return result

    return result


def assign_weak_label(
    section_name: str,
    paragraph_index: int,
    text: str = "",
    chunk_index: int = 0,
    label_rules: Dict | None = None,
) -> str | None:
    """Backward-compatible wrapper that returns only the weak label."""
    return assign_weak_label_metadata(
        section_name=section_name,
        paragraph_index=paragraph_index,
        text=text,
        chunk_index=chunk_index,
        label_rules=label_rules,
    )["weak_label"]


def _archive_paths(split: str, cache_dir: str | Path) -> Tuple[Path, str, str]:
    if split not in ARCHIVE_BY_SPLIT:
        raise ValueError(f"Unsupported split '{split}'. Expected one of {sorted(ARCHIVE_BY_SPLIT)}")

    cache_dir = Path(cache_dir)
    archive_name, url, json_name = ARCHIVE_BY_SPLIT[split]
    return cache_dir / archive_name, url, json_name


def download_qasper_archive(split: str = "train", cache_dir: str | Path = "data/raw/qasper") -> Path:
    """Download the QASPER archive for the requested split if needed."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    archive_path, url, _ = _archive_paths(split, cache_dir)
    if archive_path.exists():
        return archive_path

    urllib.request.urlretrieve(url, archive_path)
    return archive_path


def ensure_qasper_json(split: str = "train", cache_dir: str | Path = "data/raw/qasper") -> Path:
    """Extract the requested QASPER split JSON to the cache directory if needed."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    archive_path, _, json_name = _archive_paths(split, cache_dir)
    json_path = cache_dir / json_name
    if json_path.exists():
        return json_path

    download_qasper_archive(split=split, cache_dir=cache_dir)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extract(json_name, path=cache_dir)

    return json_path


def load_qasper_papers(split: str = "train", cache_dir: str | Path = "data/raw/qasper") -> List[Tuple[str, Dict]]:
    """Load QASPER paper records as `(paper_id, paper_dict)` pairs."""
    json_path = ensure_qasper_json(split=split, cache_dir=cache_dir)
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return list(data.items())


def iter_qasper_sections(paper: Dict, include_title: bool = False) -> Iterable[Tuple[str, List[str]]]:
    """Yield abstract and full-text sections in a unified structure."""
    title = normalize_whitespace(paper.get("title", ""))
    if include_title and title:
        yield "title", [title]

    abstract = normalize_whitespace(paper.get("abstract", ""))
    if abstract:
        yield "abstract", [abstract]

    for section in paper.get("full_text", []):
        section_name = section.get("section_name") or "unknown"
        paragraphs = section.get("paragraphs") or []
        cleaned_paragraphs = [normalize_whitespace(paragraph) for paragraph in paragraphs if normalize_whitespace(paragraph)]
        if cleaned_paragraphs:
            yield section_name, cleaned_paragraphs


def extract_qasper_spans(
    split: str = "train",
    max_papers: int = 100,
    min_tokens: int = 60,
    max_tokens: int = 180,
    cache_dir: str | Path = "data/raw/qasper",
    domain: str = "scientific",
    label_rules: Dict | None = None,
) -> tuple[pd.DataFrame, Dict]:
    """Build a weak-labeled span dataset from real QASPER papers."""
    papers = load_qasper_papers(split=split, cache_dir=cache_dir)
    rules = resolve_label_rules(label_rules)

    records: List[Dict] = []
    class_counts: Counter = Counter()
    confidence_counts: Counter = Counter()
    dropped_candidate_counts: Counter = Counter()
    drop_reason_counts: Counter = Counter()
    papers_with_spans = 0
    papers_scanned = 0
    next_span_id = 0

    for paper_id, paper in papers:
        if papers_with_spans >= max_papers:
            break

        papers_scanned += 1
        paper_records: List[Dict] = []

        for section_name, paragraphs in iter_qasper_sections(paper, include_title=rules.get("include_title", False)):
            for paragraph_index, paragraph in enumerate(paragraphs):
                span_source_text = paragraph
                if (
                    rules.get("mode") == "prototype"
                    and rules.get("prototype_lead_sentence_only", False)
                    and paragraph_index == 0
                    and normalize_section_name(section_name) not in {"title", "abstract", "introduction", "conclusion", "conclusions"}
                ):
                    span_source_text = first_sentence(paragraph)

                for chunk_index, (chunk_text, token_count) in enumerate(
                    chunk_text_by_tokens(span_source_text, min_tokens=min_tokens, max_tokens=max_tokens)
                ):
                    label_metadata = assign_weak_label_metadata(
                        section_name,
                        paragraph_index=paragraph_index,
                        text=chunk_text,
                        chunk_index=chunk_index,
                        label_rules=rules,
                    )
                    weak_label = label_metadata["weak_label"]
                    if weak_label is None:
                        dropped_candidate = label_metadata.get("dropped_candidate_label")
                        drop_reason = label_metadata.get("drop_reason")
                        if dropped_candidate:
                            dropped_candidate_counts[dropped_candidate] += 1
                        if drop_reason:
                            drop_reason_counts[drop_reason] += 1
                        continue

                    paper_records.append(
                        {
                            "paper_id": paper_id,
                            "domain": domain,
                            "section_name": section_name,
                            "paragraph_index": paragraph_index,
                            "chunk_index": chunk_index,
                            "span_id": f"qasper_span_{next_span_id:06d}",
                            "text": chunk_text,
                            "weak_label": weak_label,
                            "weak_label_confidence": label_metadata["weak_label_confidence"],
                            "weak_label_signal_count": label_metadata["weak_label_signal_count"],
                            "token_count": token_count,
                        }
                    )
                    class_counts[weak_label] += 1
                    confidence_counts[f"{weak_label}:{label_metadata['weak_label_confidence']}"] += 1
                    next_span_id += 1

        if paper_records:
            records.extend(paper_records)
            papers_with_spans += 1

    df = pd.DataFrame(records, columns=OUTPUT_COLUMNS)

    stats = {
        "split": split,
        "papers_scanned": papers_scanned,
        "papers_with_spans": papers_with_spans,
        "total_spans": int(len(df)),
        "class_counts": {label: int(class_counts.get(label, 0)) for label in ("macro", "meso", "micro")},
        "confidence_counts": {label: int(count) for label, count in sorted(confidence_counts.items())},
        "dropped_candidate_counts": {label: int(count) for label, count in sorted(dropped_candidate_counts.items())},
        "drop_reason_counts": {reason: int(count) for reason, count in sorted(drop_reason_counts.items())},
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "label_rules": rules,
    }

    return df, stats


def save_qasper_spans(
    output_path: str | Path = "data/processed/qasper_spans.csv",
    split: str = "train",
    max_papers: int = 100,
    min_tokens: int = 60,
    max_tokens: int = 180,
    cache_dir: str | Path = "data/raw/qasper",
    domain: str = "scientific",
    label_rules: Dict | None = None,
) -> tuple[pd.DataFrame, Dict]:
    """Extract and persist QASPER spans to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df, stats = extract_qasper_spans(
        split=split,
        max_papers=max_papers,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        cache_dir=cache_dir,
        domain=domain,
        label_rules=label_rules,
    )
    df.to_csv(output_path, index=False)
    stats["output_path"] = str(output_path)
    return df, stats
