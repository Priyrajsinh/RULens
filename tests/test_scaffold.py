"""Scaffold tests: config, logging, seeding, schemas, and import smoke tests.

Every implemented module has at least one assertion here, and every stub module
is import-smoke-tested so the package always imports cleanly.
"""

from __future__ import annotations

import importlib
import json
import logging
import random

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from src.api.schemas import (
    ActiveFeature,
    ConformalInterval,
    ForecastRequest,
    ForecastResponse,
)
from src.common.config import config_hash, load_config
from src.common.exceptions import ConfigError, RULensError
from src.common.logger import JSONFormatter, get_logger
from src.common.seeding import set_seed

ALL_MODULES = [
    "src.common.config",
    "src.common.logger",
    "src.common.exceptions",
    "src.common.seeding",
    "src.data.loaders",
    "src.models.chronos",
    "src.models.baselines",
    "src.models.hooks",
    "src.sae.architectures",
    "src.sae.training",
    "src.interp.gallery",
    "src.faithfulness.patching",
    "src.conformal.split_cp",
    "src.causal.double_ml",
    "src.api.app",
    "src.api.schemas",
    "src.ui.dashboard",
]


@pytest.mark.parametrize("module", ALL_MODULES)
def test_every_module_imports(module: str) -> None:
    assert importlib.import_module(module) is not None


# --- config -----------------------------------------------------------------


def test_load_config_defaults() -> None:
    config = load_config()
    assert config.model.chronos_variant == "amazon/chronos-t5-base"
    assert config.sae.k <= config.sae.dict_size
    assert 0 < config.conformal.alpha < 1
    assert config.seed == 1337


def test_config_hash_is_stable() -> None:
    assert config_hash(load_config()) == config_hash(load_config())


def test_load_config_missing_file_raises() -> None:
    with pytest.raises(ConfigError):
        load_config("config/does_not_exist.yaml")


def test_load_config_rejects_bad_alpha(tmp_path) -> None:
    raw = yaml.safe_load(open("config/config.yaml", encoding="utf-8"))
    raw["conformal"]["alpha"] = 1.5
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_config_error_is_rulens_error() -> None:
    assert issubclass(ConfigError, RULensError)


# --- logger -----------------------------------------------------------------


def test_logger_emits_valid_json() -> None:
    record = logging.LogRecord(
        name="rulens.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    record.run_id = "abc123"
    parsed = json.loads(JSONFormatter().format(record))
    assert parsed["message"] == "hello world"
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "rulens.test"
    assert parsed["run_id"] == "abc123"


def test_get_logger_is_idempotent() -> None:
    logger = get_logger("rulens.idempotent.test")
    handler_count = len(logger.handlers)
    again = get_logger("rulens.idempotent.test")
    assert again is logger
    assert len(again.handlers) == handler_count == 1


# --- seeding ----------------------------------------------------------------


def test_set_seed_returns_seed() -> None:
    assert set_seed(42) == 42


def test_seeding_is_deterministic() -> None:
    set_seed(123)
    py_first = [random.random() for _ in range(4)]
    np_first = np.random.rand(4).tolist()

    set_seed(123)
    py_second = [random.random() for _ in range(4)]
    np_second = np.random.rand(4).tolist()

    assert py_first == py_second
    assert np_first == np_second


# --- schemas ----------------------------------------------------------------


def test_forecast_request_defaults_alpha_to_none() -> None:
    request = ForecastRequest(engine_id="FD001-7", series=[1.0, 0.9], horizon=5)
    assert request.alpha is None


def test_forecast_request_accepts_explicit_alpha() -> None:
    request = ForecastRequest(
        engine_id="FD001-7", series=[1.0, 0.9], horizon=5, alpha=0.05
    )
    assert request.alpha == 0.05


def test_forecast_request_rejects_bad_alpha() -> None:
    with pytest.raises(ValidationError):
        ForecastRequest(engine_id="e", series=[1.0], horizon=1, alpha=0.0)


def test_conformal_interval_rejects_inverted_bounds() -> None:
    with pytest.raises(ValidationError):
        ConformalInterval(lower=[1.0, 5.0], upper=[2.0, 3.0], alpha=0.1)


def test_forecast_response_enforces_matching_lengths() -> None:
    interval = ConformalInterval(lower=[0.0], upper=[2.0], alpha=0.1)
    with pytest.raises(ValidationError):
        ForecastResponse(
            engine_id="e",
            point_forecast=[1.0, 1.0],
            interval=interval,
        )


def test_forecast_response_carries_coverage_caveat() -> None:
    interval = ConformalInterval(lower=[0.0], upper=[2.0], alpha=0.1)
    response = ForecastResponse(
        engine_id="e",
        point_forecast=[1.0],
        interval=interval,
        active_features=[ActiveFeature(feature_id=3, activation=0.7)],
    )
    assert "marginal" in response.coverage_note
