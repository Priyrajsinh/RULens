"""Fit-on-train scaling and near-constant sensor pruning for C-MAPSS.

Two scaling regimes are supported. Single-condition subsets (FD001/FD003) use a
single ``StandardScaler``. Multi-condition subsets (FD002/FD004) cycle through
six discrete operating points whose effect dwarfs the degradation signal; for
those we cluster the operating settings into regimes and fit one scaler per
regime, so the normalized sensors reflect wear rather than which operating point
the engine happened to be in. Every scaler is fit on the training rows only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .loaders import SENSOR_COLS

OP_SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]


def select_feature_cols(train: pd.DataFrame, variance_threshold: float) -> list[str]:
    """Return sensors whose train variance clears the threshold.

    Several C-MAPSS sensors are constant or near-constant and carry no
    degradation information; dropping them keeps the feature set honest.
    """
    return [c for c in SENSOR_COLS if float(train[c].var()) >= variance_threshold]


def fit_scaler(train: pd.DataFrame, feature_cols: list[str]) -> StandardScaler:
    """Fit a single ``StandardScaler`` on the training feature columns only."""
    return StandardScaler().fit(train[feature_cols].to_numpy(dtype=np.float64))


def apply_scaler(
    frame: pd.DataFrame, scaler: StandardScaler, feature_cols: list[str]
) -> pd.DataFrame:
    """Return a copy with ``feature_cols`` replaced by their scaled values."""
    out = frame.copy()
    out[feature_cols] = scaler.transform(out[feature_cols].to_numpy(dtype=np.float64))
    return out


def fit_regime_labeler(train: pd.DataFrame, n_regimes: int, seed: int) -> KMeans:
    """Cluster the three operating settings into discrete operating regimes.

    The operating points are well separated, so the partition is stable; the
    seed is recorded for reproducibility rather than to escape a poor optimum.
    """
    labeler = KMeans(n_clusters=n_regimes, random_state=seed, n_init="auto")
    labeler.fit(train[OP_SETTING_COLS].to_numpy(dtype=np.float64))
    return labeler


def assign_regimes(frame: pd.DataFrame, labeler: KMeans) -> np.ndarray:
    """Label each row with its operating regime using a fitted labeler."""
    return labeler.predict(frame[OP_SETTING_COLS].to_numpy(dtype=np.float64))


def fit_regime_scalers(
    train: pd.DataFrame, feature_cols: list[str], labeler: KMeans
) -> dict[int, StandardScaler]:
    """Fit one ``StandardScaler`` per operating regime on the training rows."""
    labels = assign_regimes(train, labeler)
    scalers: dict[int, StandardScaler] = {}
    for regime in np.unique(labels):
        rows = train.loc[labels == regime, feature_cols].to_numpy(dtype=np.float64)
        scalers[int(regime)] = StandardScaler().fit(rows)
    return scalers


def apply_regime_scalers(
    frame: pd.DataFrame,
    scalers: dict[int, StandardScaler],
    labeler: KMeans,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Scale every row by the scaler fitted for its operating regime."""
    out = frame.copy()
    labels = assign_regimes(frame, labeler)
    values = out[feature_cols].to_numpy(dtype=np.float64)
    for regime, scaler in scalers.items():
        mask = labels == regime
        if mask.any():
            values[mask] = scaler.transform(values[mask])
    out[feature_cols] = values
    return out
