"""Double-ML attribution of SAE features onto predicted RUL.

Secondary evidence only: SAE features are correlated, not randomized treatments,
so every effect size carries the unconfoundedness caveat. Activation patching
stays the primary causal method.
"""

from __future__ import annotations

from typing import Any


def estimate_feature_effect(
    features: Any, treatment_id: int, outcome: Any
) -> dict[str, float]:
    """Estimate one feature's effect on RUL; report it with the DML caveat."""
    raise NotImplementedError


UNCONFOUNDEDNESS_CAVEAT: str = (
    "SAE features are observational and correlated, not randomized treatments; "
    "Double-ML effect sizes assume unconfoundedness, which is not guaranteed "
    "here. Treat these as secondary to activation-patching evidence."
)
