"""Supervised RUL baselines: LSTM, Transformer, PatchTST.

These set the forecasting floor the foundation models are compared against on
C-MAPSS (RMSE / MAE / NASA score). Trained from config; no magic numbers here.
"""

from __future__ import annotations

from typing import Any

from ..common.config import Config


def build_baseline(name: str, config: Config) -> Any:
    """Construct an untrained baseline ('lstm' | 'transformer' | 'patchtst')."""
    raise NotImplementedError


def train_baseline(model: Any, train_data: Any, config: Config) -> Any:
    """Train a baseline and return it; the caller logs metrics to MLflow."""
    raise NotImplementedError
