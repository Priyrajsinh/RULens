"""Chronos time-series foundation-model wrapper.

The interpretability target is Chronos-Base (>=200M); Bolt-Small/Tiny are for the
deploy demo only. Device always comes from config / --device, never a hardcoded
"cuda".
"""

from __future__ import annotations

from typing import Any

from ..common.config import ModelConfig


def load_chronos(config: ModelConfig, device: str) -> Any:
    """Load a Chronos variant and place it on ``device``. No import-time load."""
    raise NotImplementedError


def forecast(model: Any, series: Any, prediction_length: int) -> Any:
    """Return the point forecast for ``series`` over ``prediction_length``."""
    raise NotImplementedError
