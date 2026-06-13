"""Split conformal prediction for RUL, with coverage and sharpness metrics.

The baseline calibrated interval. Its guarantee is marginal and assumes
exchangeability, which engine degradation breaks — the motivation for CPTC.
Always report coverage with that qualifier.
"""

from __future__ import annotations

from typing import Any

from ..common.config import ConformalConfig


def calibrate(residuals: Any, alpha: float) -> float:
    """Return the conformal quantile of calibration residuals at level alpha."""
    raise NotImplementedError


def predict_interval(point_forecast: Any, quantile: float) -> tuple[Any, Any]:
    """Build (lower, upper) by adding +/- the conformal quantile to the forecast."""
    raise NotImplementedError


def coverage_and_width(
    lower: Any, upper: Any, truth: Any, config: ConformalConfig
) -> dict[str, float]:
    """Return empirical (marginal) coverage and mean interval width."""
    raise NotImplementedError
