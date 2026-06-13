"""Forward-hook activation capture for the residual stream.

Registers hooks on the layers named in ``ModelConfig.layers_to_hook``, runs the
model, and returns per-layer activations to be cached (DVC) for SAE training.
The activation corpus is the substrate every interpretability claim rests on.
"""

from __future__ import annotations

from typing import Any

from ..common.config import ModelConfig


def register_activation_hooks(model: Any, config: ModelConfig) -> list[Any]:
    """Attach forward hooks to the configured layers; return their handles."""
    raise NotImplementedError


def capture_activations(model: Any, inputs: Any, config: ModelConfig) -> dict[int, Any]:
    """Run a forward pass and return ``{layer_index: activation_tensor}``."""
    raise NotImplementedError
