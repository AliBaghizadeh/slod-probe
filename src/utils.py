"""Prototype-compatible utility wrapper.

This file exists to mirror the owner-requested prototype layout while still
allowing imports such as ``src.utils.seed`` to resolve to the active package
implementation in ``src/utils/``.
"""

from __future__ import annotations

import json
from pathlib import Path

__path__ = [str(Path(__file__).with_name("utils"))]


def prototype_path_map() -> dict[str, str]:
    """Return the mapping from prototype expectations to active repo paths."""
    return {
        "prototype_spans_dir": "data/spans/",
        "prototype_embeddings_dir": "embeddings/",
        "active_scibert_cache": "data/cache_refined_qasper_slod_scibert/",
        "active_scibert_controlled_cache": "data/cache_refined_qasper_slod_scibert_length_controlled/",
        "active_minilm_cache": "data/cache_refined_qasper_slod_minilm/",
        "active_minilm_controlled_cache": "data/cache_refined_qasper_slod_minilm_length_controlled/",
        "prototype_results_dir": "results/prototype/",
    }


if __name__ == "__main__":
    print(json.dumps(prototype_path_map(), indent=2))
