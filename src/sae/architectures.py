"""Sparse-autoencoder architectures: TopK and JumpReLU.

TopK is the default (deterministic sparsity = k active latents); JumpReLU is the
Day-13 comparison for the sparsity-fidelity trade-off. Both expose the same
encode/decode contract so the training and interp code is architecture-agnostic.
"""

from __future__ import annotations

from typing import Any

from ..common.config import SAEConfig


def build_sae(config: SAEConfig, input_dim: int) -> Any:
    """Construct an untrained SAE of the configured architecture."""
    raise NotImplementedError


def encode(sae: Any, activations: Any) -> Any:
    """Map activations to sparse feature codes."""
    raise NotImplementedError


def decode(sae: Any, codes: Any) -> Any:
    """Reconstruct activations from sparse feature codes."""
    raise NotImplementedError
