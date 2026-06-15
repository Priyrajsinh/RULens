"""Tests for the C-MAPSS data layer: loaders, scaling, windowing, and build.

Everything here runs on tiny synthetic frames written into ``tmp_path`` so the
suite passes in CI where ``data/`` is absent. The one test that touches the real
NASA files skips automatically when they are not present.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from src.common.config import Config, load_config
from src.common.exceptions import DataError
from src.data.build import (
    build_arrays,
    build_parser,
    main,
    read_dvc_md5,
    run_build,
    update_summary,
)
from src.data.loaders import (
    COLUMNS,
    SENSOR_COLS,
    add_piecewise_rul,
    build_dataset,
    load_cmapss_subset,
    make_windows,
)
from src.data.scaling import (
    apply_regime_scalers,
    apply_scaler,
    fit_regime_labeler,
    fit_regime_scalers,
    fit_scaler,
    select_feature_cols,
)

SENSOR_NAMES = SENSOR_COLS


# --- synthetic fixtures ------------------------------------------------------


def _toy(n_cycles: int = 50, unit: int = 1) -> pd.DataFrame:
    """One healthy-to-failure engine with increasing sensor readings."""
    rows = [
        {
            "unit": unit,
            "cycle": c,
            "op_setting_1": 0.0,
            "op_setting_2": 0.0,
            "op_setting_3": 100.0,
            "sensor_1": 100.0,  # constant -> should be dropped
            **{f"sensor_{i}": float(i) + 0.1 * c for i in range(2, 22)},
        }
        for c in range(1, n_cycles + 1)
    ]
    return pd.DataFrame(rows)


def _write_subset(
    dir_path: Path,
    subset: str,
    *,
    n_units: int = 4,
    base_len: int = 40,
    n_op_points: int = 1,
    seed: int = 0,
) -> None:
    """Write a synthetic ``train_<subset>.txt`` in the raw C-MAPSS layout."""
    rng = np.random.default_rng(seed)
    op_points = rng.normal(size=(max(n_op_points, 1), 3)) * 50.0
    lines: list[str] = []
    for unit in range(1, n_units + 1):
        for cycle in range(1, base_len + unit + 1):
            op = op_points[(cycle - 1) % len(op_points)] + rng.normal(
                scale=0.01, size=3
            )
            sensors = [
                100.0 if s == 1 else 50.0 + 0.1 * cycle * s + rng.normal(scale=0.5)
                for s in range(1, 22)
            ]
            values = [str(unit), str(cycle)] + [f"{v:.4f}" for v in (*op, *sensors)]
            lines.append(" ".join(values))
    (dir_path / f"train_{subset}.txt").write_text("\n".join(lines) + "\n", "utf-8")


def _cfg(tmp_path: Path, **data_overrides: object) -> Config:
    """A Config pointing at a temp raw dir, processed dir, and MLflow store."""
    base = load_config()
    data = base.data.model_copy(
        update={
            "cmapss_path": str(tmp_path),
            "processed_dir": str(tmp_path / "processed"),
            **data_overrides,
        }
    )
    mlflow_cfg = base.mlflow.model_copy(
        update={"tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}"}
    )
    return base.model_copy(update={"data": data, "mlflow": mlflow_cfg})


def _write_config(tmp_path: Path) -> Path:
    """Write a config.yaml pointing at the temp raw/processed dirs for the CLI."""
    raw = yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))
    raw["data"]["cmapss_path"] = str(tmp_path)
    raw["data"]["processed_dir"] = str(tmp_path / "processed")
    raw["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path / 'mlflow.db'}"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return cfg_path


# --- loaders -----------------------------------------------------------------


def test_load_subset_schema(tmp_path: Path) -> None:
    _write_subset(tmp_path, "FD001")
    frame = load_cmapss_subset(tmp_path, "FD001")
    assert list(frame.columns) == COLUMNS
    assert frame["unit"].dtype == int and frame["cycle"].dtype == int
    assert frame["unit"].nunique() == 4


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DataError):
        load_cmapss_subset(tmp_path, "FD099")


def test_rul_clipped_and_monotone() -> None:
    df = add_piecewise_rul(_toy(), rul_clip=125)
    assert df["rul"].max() <= 125
    assert df["rul"].is_monotonic_decreasing
    assert df["rul"].iloc[-1] == 0.0


def test_rul_clip_is_applied() -> None:
    # A 200-cycle engine must hit the clip early instead of starting at 199.
    df = add_piecewise_rul(_toy(n_cycles=200), rul_clip=125)
    assert df["rul"].iloc[0] == 125.0


def test_window_shapes() -> None:
    df = add_piecewise_rul(_toy(), rul_clip=125)
    x, y = make_windows(df, 30, 1, SENSOR_NAMES)
    assert x.shape[1:] == (30, 21)
    assert len(x) == len(y)
    # 50 cycles, window 30, stride 1 -> 21 windows.
    assert len(x) == 21


def test_window_stride_reduces_count() -> None:
    df = add_piecewise_rul(_toy(), rul_clip=125)
    _, y1 = make_windows(df, 30, 1, SENSOR_NAMES)
    _, y2 = make_windows(df, 30, 2, SENSOR_NAMES)
    assert len(y1) == 21
    assert len(y2) == 11  # floor((50-30)/2)+1


def test_short_unit_is_left_padded() -> None:
    df = add_piecewise_rul(_toy(n_cycles=10), rul_clip=125)
    x, y = make_windows(df, 30, 1, SENSOR_NAMES)
    assert x.shape == (1, 30, 21)
    # Last row of the window is the engine's final (lowest) RUL.
    assert y[0] == 0.0


# --- scaling -----------------------------------------------------------------


def test_select_feature_cols_drops_constant() -> None:
    kept = select_feature_cols(_toy(), variance_threshold=1.0e-6)
    assert "sensor_1" not in kept
    assert "sensor_2" in kept
    assert len(kept) == 20


def test_scaler_is_fit_on_train_only() -> None:
    train = _toy()
    cols = ["sensor_2", "sensor_3"]
    scaler = fit_scaler(train, cols)
    scaled_train = apply_scaler(train, scaler, cols)
    assert np.allclose(scaled_train[cols].mean().to_numpy(), 0.0, atol=1e-6)
    # A shifted hold-out frame must not come out mean-zero (no leakage).
    shifted = train.copy()
    shifted[cols] = shifted[cols] + 1000.0
    assert apply_scaler(shifted, scaler, cols)["sensor_2"].mean() > 1.0


def test_per_regime_scaling_standardizes_within_regime() -> None:
    rows = []
    for unit, op_value, center in [(1, 0.0, 10.0), (2, 100.0, 500.0)]:
        for cycle in range(1, 31):
            rows.append(
                {
                    "unit": unit,
                    "cycle": cycle,
                    "op_setting_1": op_value,
                    "op_setting_2": op_value,
                    "op_setting_3": op_value,
                    **{f"sensor_{i}": center + cycle for i in range(1, 22)},
                }
            )
    frame = pd.DataFrame(rows)
    cols = ["sensor_2", "sensor_3"]
    labeler = fit_regime_labeler(frame, n_regimes=2, seed=0)
    scalers = fit_regime_scalers(frame, cols, labeler)
    scaled = apply_regime_scalers(frame, scalers, labeler, cols)
    assert len(scalers) == 2
    for unit in (1, 2):
        within = scaled.loc[scaled["unit"] == unit, cols].mean().to_numpy()
        assert np.allclose(within, 0.0, atol=1e-6)


# --- build -------------------------------------------------------------------


def test_build_arrays_single_regime(tmp_path: Path) -> None:
    _write_subset(tmp_path, "FD001")
    cfg = _cfg(tmp_path)
    art = build_arrays(cfg.data, "FD001", cfg.seed)
    assert art.x.shape[1] == cfg.data.window_size
    assert art.x.shape[2] == len(art.feature_cols)
    assert "sensor_1" in art.dropped_sensors
    assert art.scaler_payload["kind"] == "global"
    assert art.n_units == 4


def test_build_arrays_multi_regime(tmp_path: Path) -> None:
    _write_subset(tmp_path, "FD002", n_op_points=2, seed=1)
    cfg = _cfg(tmp_path, operating_regimes=2)
    art = build_arrays(cfg.data, "FD002", cfg.seed)
    assert art.scaler_payload["kind"] == "regime"
    assert len(art.scaler_payload["scalers"]) == 2


def test_build_arrays_rejects_overzealous_threshold(tmp_path: Path) -> None:
    _write_subset(tmp_path, "FD001")
    cfg = _cfg(tmp_path, variance_threshold=1.0e30)
    with pytest.raises(DataError):
        build_arrays(cfg.data, "FD001", cfg.seed)


def test_run_build_persists_artifacts(tmp_path: Path) -> None:
    _write_subset(tmp_path, "FD001")
    cfg = _cfg(tmp_path)
    meta = run_build(cfg, "FD001", use_dvc=False, log_mlflow=False)
    out_dir = Path(cfg.data.processed_dir) / "FD001"
    for name in ("X.npy", "y.npy", "scaler.joblib", "meta.json"):
        assert (out_dir / name).exists()
    x = np.load(out_dir / "X.npy")
    assert meta["n_windows"] == x.shape[0]
    assert meta["seed"] == cfg.seed
    payload = joblib.load(out_dir / "scaler.joblib")
    assert payload["kind"] == "global"


def test_build_dataset_contract(tmp_path: Path) -> None:
    _write_subset(tmp_path, "FD001")
    cfg = _cfg(tmp_path)
    x, y = build_dataset(cfg.data, "FD001", cfg.seed)
    assert x.ndim == 3 and len(x) == len(y)


def test_update_summary_merges_subsets(tmp_path: Path) -> None:
    path = tmp_path / "reports" / "summary.json"
    update_summary(path, "FD001", {"n_windows": 10})
    update_summary(path, "FD002", {"n_windows": 20})
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert set(summary) == {"FD001", "FD002"}
    assert summary["FD002"]["n_windows"] == 20


def test_read_dvc_md5(tmp_path: Path) -> None:
    dvc_file = tmp_path / "X.npy.dvc"
    dvc_file.write_text("outs:\n- md5: abc123\n  path: X.npy\n", encoding="utf-8")
    assert read_dvc_md5(dvc_file) == "abc123"


# --- CLI ---------------------------------------------------------------------


def test_build_parser_parses_args() -> None:
    args = build_parser().parse_args(["--subset", "FD003", "--no-dvc"])
    assert args.subset == "FD003"
    assert args.no_dvc is True
    assert args.config == "config/config.yaml"


def test_main_cli_builds_and_writes_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_subset(tmp_path, "FD001")
    cfg_path = _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    main(["--subset", "FD001", "--config", str(cfg_path), "--no-dvc"])
    assert (tmp_path / "processed" / "FD001" / "X.npy").exists()
    summary = json.loads(
        (tmp_path / "reports" / "data_build_summary.json").read_text("utf-8")
    )
    assert "FD001" in summary


# --- real data (skips in CI) -------------------------------------------------


def test_real_fd001_builds() -> None:
    cfg = load_config()
    if not (Path(cfg.data.cmapss_path) / "train_FD001.txt").exists():
        pytest.skip("C-MAPSS raw files not present")
    art = build_arrays(cfg.data, "FD001", cfg.seed)
    assert art.x.shape[0] > 0
    assert art.x.shape[1] == cfg.data.window_size
    # FD001 sensors 1/5/10/16/18/19 are flat and must be pruned.
    assert "sensor_1" in art.dropped_sensors
