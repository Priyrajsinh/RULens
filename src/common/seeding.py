"""Deterministic seeding for reproducible runs.

set_seed seeds Python, NumPy, and (when installed) PyTorch + CUDA, and can switch
PyTorch into deterministic-algorithm mode. NumPy and Torch are imported lazily,
so there's no import-time cost or side effect. Returns the seed so callers can
log it alongside the run.
"""

from __future__ import annotations

import os
import random


def set_seed(seed: int, deterministic: bool = True) -> int:
    """Seed every available RNG and return the seed.

    Args:
        seed: The integer seed applied to Python, NumPy, and Torch RNGs.
        deterministic: When True and Torch is installed, request deterministic
            algorithms and disable cuDNN autotuning. Nondeterministic ops fall
            back with a warning rather than raising (``warn_only=True``).

    Returns:
        The same ``seed``, for convenient logging into an MLflow run.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            # Required for deterministic CUDA matmul; harmless on CPU.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
