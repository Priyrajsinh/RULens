"""Load and validate the YAML config into typed Pydantic models.

A malformed or incomplete config fails at startup instead of halfway through a
run. File I/O happens in load_config, never at import.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .exceptions import ConfigError

DEFAULT_CONFIG_PATH = Path("config/config.yaml")

# Leading hex chars kept from the full sha256 for a compact config fingerprint.
_HASH_DISPLAY_CHARS = 16

# Reject unknown keys (so a typo in the YAML is a hard error) and free up the
# "model" field name from Pydantic's protected namespace.
_STRICT = ConfigDict(extra="forbid", protected_namespaces=())


class ModelConfig(BaseModel):
    """Forecaster + activation-capture settings."""

    model_config = _STRICT

    chronos_variant: str
    layers_to_hook: list[int]
    context_length: int = Field(gt=0)
    prediction_length: int = Field(gt=0)

    @field_validator("layers_to_hook")
    @classmethod
    def _check_layers(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("layers_to_hook must not be empty")
        if any(layer < 0 for layer in value):
            raise ValueError("layer indices must be non-negative")
        return value


class SAEConfig(BaseModel):
    """Sparse-autoencoder training settings."""

    model_config = _STRICT

    dict_size: int = Field(gt=0)
    k: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    batch_size: int = Field(gt=0)
    architecture: Literal["topk", "jumprelu"]

    @model_validator(mode="after")
    def _k_within_dict(self) -> "SAEConfig":
        if self.k > self.dict_size:
            raise ValueError("k must not exceed dict_size")
        return self


class ConformalConfig(BaseModel):
    """Conformal-prediction settings (coverage is marginal under exchangeability)."""

    model_config = _STRICT

    alpha: float = Field(gt=0, lt=1)
    method: Literal["split_cp", "enbpi", "spci", "cptc"]


class DataConfig(BaseModel):
    """Dataset paths and windowing parameters."""

    model_config = _STRICT

    cmapss_path: str
    bosch_path: str
    processed_dir: str
    window_size: int = Field(gt=0)
    window_stride: int = Field(gt=0)
    rul_clip: int = Field(gt=0)
    variance_threshold: float = Field(gt=0)
    operating_regimes: int = Field(gt=0)
    multi_regime_subsets: list[str]


class MLflowConfig(BaseModel):
    """Experiment-tracking settings."""

    model_config = _STRICT

    tracking_uri: str
    experiment_name: str


class Config(BaseModel):
    """Top-level validated RULens configuration."""

    model_config = _STRICT

    seed: int
    device: str
    model: ModelConfig
    sae: SAEConfig
    conformal: ConformalConfig
    data: DataConfig
    mlflow: MLflowConfig


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load and validate the YAML config into a :class:`Config`.

    Raises:
        ConfigError: If the file is missing, is not valid YAML, does not parse
            to a mapping, or fails schema validation.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(raw).__name__}")
    try:
        return Config(**raw)
    except ValidationError as exc:
        raise ConfigError(f"Config validation failed:\n{exc}") from exc


def config_hash(config: Config) -> str:
    """Return a short, stable hash of the config for run provenance."""
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:_HASH_DISPLAY_CHARS]
