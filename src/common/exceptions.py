"""Exception hierarchy for RULens.

A single base (:class:`RULensError`) so callers can catch every project-raised
error with one ``except``, plus one subclass per subsystem so failures are
attributable to a layer (config, data, model, SAE, conformal, ...).
"""

from __future__ import annotations


class RULensError(Exception):
    """Base class for every error raised by RULens code."""


class ConfigError(RULensError):
    """Raised when configuration is missing, malformed, or fails validation."""


class DataError(RULensError):
    """Raised for dataset loading, labeling, windowing, or split failures."""


class ModelError(RULensError):
    """Raised for forecaster / wrapper loading or inference failures."""


class ActivationError(ModelError):
    """Raised when activation capture (forward hooks) misbehaves."""


class SAEError(RULensError):
    """Raised for sparse-autoencoder training or feature-extraction failures."""


class FaithfulnessError(RULensError):
    """Raised for activation-patching / ablation intervention failures."""


class ConformalError(RULensError):
    """Raised for conformal calibration or coverage-evaluation failures."""


class CausalError(RULensError):
    """Raised for Double-ML attribution failures (secondary evidence only)."""
