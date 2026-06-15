"""Build a processed C-MAPSS subset and record its provenance.

Pipeline: load -> piecewise RUL -> drop near-constant sensors -> fit-on-train
scaling -> sliding windows -> persist ``X.npy`` / ``y.npy`` / ``scaler.joblib``.
The arrays are versioned with DVC and every build is logged as an MLflow run so
each reported dataset statistic traces back to a run id.

Run as a module::

    python -m src.data.build --subset FD001 --config config/config.yaml

``mlflow`` and ``dvc`` are imported lazily (and skipped gracefully when absent)
so importing this module stays light for CI and downstream code.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

from ..common.config import Config, DataConfig, config_hash, load_config
from ..common.exceptions import DataError
from ..common.logger import get_logger
from ..common.seeding import set_seed
from .loaders import add_piecewise_rul, load_cmapss_subset, make_windows
from .scaling import (
    SENSOR_COLS,
    apply_regime_scalers,
    apply_scaler,
    fit_regime_labeler,
    fit_regime_scalers,
    fit_scaler,
    select_feature_cols,
)

_LOGGER = get_logger("rulens.data.build")
_ARRAY_NAMES = ("X.npy", "y.npy", "scaler.joblib")


@dataclass
class BuildArtifacts:
    """Processed arrays plus the fitted scaling payload and provenance counts."""

    x: np.ndarray
    y: np.ndarray
    feature_cols: list[str]
    dropped_sensors: list[str]
    scaler_payload: dict[str, Any]
    n_units: int


def build_arrays(config: DataConfig, subset: str, seed: int) -> BuildArtifacts:
    """Load a subset and produce scaled sliding-window arrays + scaling payload.

    FD002/FD004 are scaled per operating regime; the others use one scaler. The
    returned ``scaler_payload`` is everything needed to reproduce the transform
    on unseen data (the scaler(s), the kept/dropped sensor lists, and the regime
    labeler when applicable).
    """
    frame = load_cmapss_subset(config.cmapss_path, subset)
    labeled = add_piecewise_rul(frame, config.rul_clip)
    feature_cols = select_feature_cols(labeled, config.variance_threshold)
    if not feature_cols:
        raise DataError(
            f"Every sensor was dropped for {subset}; variance_threshold too high"
        )
    dropped = [c for c in SENSOR_COLS if c not in feature_cols]

    if subset in config.multi_regime_subsets:
        labeler = fit_regime_labeler(labeled, config.operating_regimes, seed)
        scalers = fit_regime_scalers(labeled, feature_cols, labeler)
        scaled = apply_regime_scalers(labeled, scalers, labeler, feature_cols)
        payload: dict[str, Any] = {
            "kind": "regime",
            "feature_cols": feature_cols,
            "dropped_sensors": dropped,
            "labeler": labeler,
            "scalers": scalers,
        }
    else:
        scaler = fit_scaler(labeled, feature_cols)
        scaled = apply_scaler(labeled, scaler, feature_cols)
        payload = {
            "kind": "global",
            "feature_cols": feature_cols,
            "dropped_sensors": dropped,
            "scaler": scaler,
        }

    x, y = make_windows(scaled, config.window_size, config.window_stride, feature_cols)
    return BuildArtifacts(
        x=x,
        y=y,
        feature_cols=feature_cols,
        dropped_sensors=dropped,
        scaler_payload=payload,
        n_units=int(labeled["unit"].nunique()),
    )


def read_dvc_md5(dvc_path: Path) -> str:
    """Return the content hash DVC recorded for a tracked file (its version)."""
    spec = yaml.safe_load(dvc_path.read_text(encoding="utf-8"))
    return str(spec["outs"][0]["md5"])


def update_summary(summary_path: Path, subset: str, meta: dict[str, Any]) -> None:
    """Merge one subset's build metadata into the committed summary report."""
    summary: dict[str, Any] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary[subset] = meta
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_build(
    config: Config,
    subset: str,
    use_dvc: bool = True,
    log_mlflow: bool = True,
) -> dict[str, Any]:
    """Build, persist, version, and log one subset; return its metadata dict.

    ``use_dvc`` and ``log_mlflow`` are disabled by tests so the pipeline can run
    hermetically without DVC, MLflow, or network access.
    """
    set_seed(config.seed)
    artifacts = build_arrays(config.data, subset, config.seed)

    out_dir = Path(config.data.processed_dir) / subset
    out_dir.mkdir(parents=True, exist_ok=True)
    x_path, y_path, scaler_path = (out_dir / name for name in _ARRAY_NAMES)
    np.save(x_path, artifacts.x)
    np.save(y_path, artifacts.y)
    joblib.dump(artifacts.scaler_payload, scaler_path)

    meta: dict[str, Any] = {
        "subset": subset,
        "seed": config.seed,
        "config_hash": config_hash(config),
        "git_commit": _git_commit(),
        "n_units": artifacts.n_units,
        "n_windows": int(artifacts.x.shape[0]),
        "window_size": config.data.window_size,
        "window_stride": config.data.window_stride,
        "rul_clip": config.data.rul_clip,
        "n_features": len(artifacts.feature_cols),
        "feature_cols": artifacts.feature_cols,
        "dropped_sensors": artifacts.dropped_sensors,
        "scaling": artifacts.scaler_payload["kind"],
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    if use_dvc:
        meta["dvc_versions"] = {name: _dvc_add(out_dir / name) for name in _ARRAY_NAMES}
    if log_mlflow:
        meta["mlflow_run_id"] = _log_mlflow_run(config, meta)
    return meta


def _git_commit() -> str:  # pragma: no cover - shells out to git
    """Return the current commit hash, or ``"unknown"`` outside a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _dvc_add(path: Path) -> str:  # pragma: no cover - shells out to dvc
    """Track ``path`` with DVC and return the recorded content hash."""
    subprocess.run(
        [sys.executable, "-m", "dvc", "add", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return read_dvc_md5(path.with_suffix(path.suffix + ".dvc"))


def _log_mlflow_run(  # pragma: no cover - only in the real (non-CI) data build
    config: Config, meta: dict[str, Any]
) -> str | None:
    """Log the build as an MLflow run; skip cleanly if MLflow is absent."""
    try:
        import mlflow
    except ImportError:
        _LOGGER.warning("mlflow not installed; skipping run", extra=meta)
        return None
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)
    with mlflow.start_run(run_name=f"data-build-{meta['subset']}") as run:
        mlflow.log_params(
            {
                k: meta[k]
                for k in (
                    "subset",
                    "seed",
                    "config_hash",
                    "git_commit",
                    "window_size",
                    "window_stride",
                    "rul_clip",
                    "scaling",
                )
            }
        )
        mlflow.log_metrics(
            {k: float(meta[k]) for k in ("n_units", "n_windows", "n_features")}
        )
        mlflow.log_dict(
            {
                "feature_cols": meta["feature_cols"],
                "dropped_sensors": meta["dropped_sensors"],
                "dvc_versions": meta.get("dvc_versions", {}),
            },
            "dataset.json",
        )
        return run.info.run_id


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the data-build entry point."""
    parser = argparse.ArgumentParser(description="Build a processed C-MAPSS subset.")
    parser.add_argument(
        "--subset",
        required=True,
        choices=["FD001", "FD002", "FD003", "FD004"],
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--no-dvc",
        action="store_true",
        help="skip dvc add (for a quick local smoke build)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Build one subset from the command line and update the summary report."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    meta = run_build(config, args.subset, use_dvc=not args.no_dvc)
    update_summary(Path("reports/data_build_summary.json"), args.subset, meta)
    _LOGGER.info(
        "data build complete",
        extra={
            "subset": args.subset,
            "n_units": meta["n_units"],
            "n_windows": meta["n_windows"],
            "mlflow_run_id": meta.get("mlflow_run_id"),
        },
    )


if __name__ == "__main__":
    main()
