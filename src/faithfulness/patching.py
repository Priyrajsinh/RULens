"""Intervention-based faithfulness: activation patching and ablation.

The primary causal evidence: intervening on the model is a real intervention,
unlike a correlational probe. Produces faithfulness curves for the report.
"""

from __future__ import annotations

from typing import Any


def patch_feature(
    model: Any, sae: Any, feature_id: int, inputs: Any, value: float
) -> Any:
    """Set ``feature_id`` to ``value`` mid-forward; return the patched forecast."""
    raise NotImplementedError


def ablate_feature(model: Any, sae: Any, feature_id: int, inputs: Any) -> Any:
    """Zero out ``feature_id`` during the forward pass; return the forecast."""
    raise NotImplementedError


def faithfulness_curve(
    model: Any, sae: Any, inputs: Any, feature_ids: list[int]
) -> dict[int, float]:
    """Measure forecast change as features are progressively intervened on."""
    raise NotImplementedError
