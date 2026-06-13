"""SAE training harness on cached Chronos-Base activations (sae_lens, TopK).

Runs the dict-size x sparsity sweep and logs reconstruction MSE, sparsity, and
dead/dense feature counts. A collapsed dictionary is reported as a finding, not
hidden.
"""

from __future__ import annotations

from typing import Any

from ..common.config import Config


def train_sae(config: Config, activations: Any) -> Any:
    """Train one SAE; return the trained model. Caller logs the MLflow run."""
    raise NotImplementedError


def feature_statistics(sae: Any, activations: Any) -> dict[str, float]:
    """Return reconstruction MSE plus dead- and dense-feature fractions."""
    raise NotImplementedError
