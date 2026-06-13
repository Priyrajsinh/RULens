"""Feature gallery: max-activating sequences and monosemanticity scoring.

Every interpretable-feature claim is backed by max-activating examples and an
interpretability score. Produces both, plus the Neuronpedia-style HTML report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def max_activating_sequences(
    sae: Any, activations: Any, feature_id: int, top_n: int
) -> list[Any]:
    """Return the ``top_n`` sequences that most activate ``feature_id``."""
    raise NotImplementedError


def monosemanticity_score(sae: Any, feature_id: int, activations: Any) -> float:
    """Score how consistently one feature tracks a single interpretable concept."""
    raise NotImplementedError


def write_feature_gallery(sae: Any, activations: Any, out_path: str | Path) -> Path:
    """Render the feature gallery HTML to ``out_path`` and return the path."""
    raise NotImplementedError
