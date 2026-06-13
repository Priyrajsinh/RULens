"""Request/response schemas for the inference API.

A response bundles the point RUL forecast, its conformal interval, and the SAE
features active for this engine. The coverage caveat lives in the schema, so it
travels with every response.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_STRICT = ConfigDict(extra="forbid")

_COVERAGE_NOTE = (
    "Coverage is marginal and assumes exchangeability of calibration and test "
    "residuals; engine degradation can violate this, so realized coverage may "
    "drift under regime shift."
)


def _all_finite(values: list[float], field: str) -> list[float]:
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"{field} must contain only finite numbers")
    return values


class ForecastRequest(BaseModel):
    """A request to forecast remaining useful life for one engine."""

    model_config = _STRICT

    engine_id: str = Field(min_length=1)
    series: list[float] = Field(min_length=1)
    horizon: int = Field(gt=0)
    # None means "use the server's configured miscoverage level"
    # (config.yaml conformal.alpha); a client may override it per request.
    alpha: float | None = Field(default=None, gt=0, lt=1)

    @field_validator("series")
    @classmethod
    def _series_finite(cls, value: list[float]) -> list[float]:
        return _all_finite(value, "series")


class ConformalInterval(BaseModel):
    """A per-step prediction interval at miscoverage level ``alpha``."""

    model_config = _STRICT

    lower: list[float]
    upper: list[float]
    alpha: float = Field(gt=0, lt=1)

    @field_validator("lower", "upper")
    @classmethod
    def _bounds_finite(cls, value: list[float]) -> list[float]:
        return _all_finite(value, "interval bound")

    @model_validator(mode="after")
    def _check_ordering(self) -> "ConformalInterval":
        if len(self.lower) != len(self.upper):
            raise ValueError("lower and upper must have equal length")
        if any(lo > hi for lo, hi in zip(self.lower, self.upper)):
            raise ValueError("each lower bound must not exceed its upper bound")
        return self


class ActiveFeature(BaseModel):
    """An SAE feature that fired for this engine, with its activation."""

    model_config = _STRICT

    feature_id: int = Field(ge=0)
    activation: float
    description: str | None = None


class ForecastResponse(BaseModel):
    """Point RUL forecast plus calibrated interval and active features."""

    model_config = _STRICT

    engine_id: str = Field(min_length=1)
    point_forecast: list[float] = Field(min_length=1)
    interval: ConformalInterval
    active_features: list[ActiveFeature] = Field(default_factory=list)
    coverage_note: str = _COVERAGE_NOTE

    @field_validator("point_forecast")
    @classmethod
    def _forecast_finite(cls, value: list[float]) -> list[float]:
        return _all_finite(value, "point_forecast")

    @model_validator(mode="after")
    def _lengths_match(self) -> "ForecastResponse":
        n = len(self.point_forecast)
        if len(self.interval.lower) != n or len(self.interval.upper) != n:
            raise ValueError("interval bounds must match the point_forecast length")
        return self
