"""C-MAPSS / Bosch CNC loading, RUL labeling, and windowing.

Implemented on Day 1. Signatures are fixed here so downstream code (models,
SAE, conformal) can be written against a stable data contract. Heavy array
types are annotated as ``Any`` to keep this module import-light.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..common.config import DataConfig


def load_cmapss_subset(path: str | Path, subset: str) -> Any:
    """Load one C-MAPSS subset (FD001-FD004) as a tidy run-indexed frame."""
    raise NotImplementedError


def add_piecewise_rul(frame: Any, rul_clip: int) -> Any:
    """Attach a piecewise-linear RUL target, capped at ``rul_clip`` cycles."""
    raise NotImplementedError


def make_windows(frame: Any, window_size: int, stride: int) -> tuple[Any, Any]:
    """Slide fixed-length windows over each run; return (windows, targets)."""
    raise NotImplementedError


def build_dataset(config: DataConfig, subset: str) -> tuple[Any, Any]:
    """End-to-end: load -> label -> window for a C-MAPSS subset."""
    raise NotImplementedError
