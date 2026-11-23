"""Deterministic seeding for reproducible experiments."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - GPU optional
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass
