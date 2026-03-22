"""Seeding and reproducibility utilities."""

import random
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Sets seeds for random, numpy, and torch (if available).

    Args:
        seed: random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_seed() -> int:
    """
    Get current seed value (default seed used by set_seed).

    Returns:
        seed integer.
    """
    return 42
