"""C-MAPSS loading, RUL labeling, and sliding-window construction.

The raw NASA C-MAPSS files are space-separated with 26 columns (unit, cycle,
three operating settings, 21 sensors) and two trailing empty fields. These
helpers turn one subset into a tidy frame, attach the piecewise-linear RUL
target, and cut per-engine windows. ``build_dataset`` ties the full pipeline
together; it defers the scaling-heavy import so this module stays light to
import for the rest of the codebase.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..common.config import DataConfig
from ..common.exceptions import DataError

# C-MAPSS records a fixed 21 sensor channels per operational cycle (NASA schema).
N_SENSORS = 21
SENSOR_COLS = [f"sensor_{i}" for i in range(1, N_SENSORS + 1)]
COLUMNS = [
    "unit",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
] + SENSOR_COLS


def load_cmapss_subset(path: str | Path, subset: str) -> pd.DataFrame:
    """Load ``train_<subset>.txt`` into a tidy, run-indexed DataFrame."""
    file_path = Path(path) / f"train_{subset}.txt"
    if not file_path.exists():
        raise DataError(f"C-MAPSS file not found: {file_path}")
    # Whitespace-separated; the two trailing empty fields are sliced off.
    frame = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
    frame = frame.iloc[:, : len(COLUMNS)]
    frame.columns = COLUMNS
    return frame.astype({"unit": int, "cycle": int})


def add_piecewise_rul(frame: pd.DataFrame, rul_clip: int) -> pd.DataFrame:
    """Attach piecewise-linear RUL ``min(max_cycle(unit) - cycle, rul_clip)``.

    The target is flat at ``rul_clip`` while the engine is healthy and decreases
    linearly to zero as it approaches failure — the standard C-MAPSS convention.
    """
    out = frame.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = (max_cycle - out["cycle"]).clip(upper=rul_clip).astype(float)
    return out


def make_windows(
    frame: pd.DataFrame,
    window_size: int,
    stride: int,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Cut fixed-length sliding windows per engine.

    Units shorter than ``window_size`` are left-padded with a copy of their
    first row so every engine yields at least one window. Returns
    ``(X[n, window_size, n_features], y[n])`` where each target is the RUL at the
    last cycle of its window.
    """
    windows: list[np.ndarray] = []
    targets: list[float] = []
    for _, group in frame.sort_values(["unit", "cycle"]).groupby("unit"):
        feats = group[feature_cols].to_numpy(dtype=np.float32)
        rul = group["rul"].to_numpy(dtype=np.float32)
        if len(group) < window_size:
            pad_len = window_size - len(group)
            feats = np.concatenate([np.repeat(feats[:1], pad_len, axis=0), feats])
            rul = np.concatenate([np.repeat(rul[:1], pad_len), rul])
        for end in range(window_size, len(feats) + 1, stride):
            windows.append(feats[end - window_size : end])
            targets.append(rul[end - 1])
    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


def build_dataset(
    config: DataConfig, subset: str, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """End-to-end load -> RUL -> fit-on-train scaling -> window for one subset.

    This is the stable contract downstream code (models, SAE, conformal) builds
    against. The scaling/build machinery is imported lazily so importing this
    module does not pull in scikit-learn.
    """
    from .build import build_arrays

    artifacts = build_arrays(config, subset, seed)
    return artifacts.x, artifacts.y
