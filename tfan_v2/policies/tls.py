"""Task facing API for topology assisted modules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:  # pragma: no cover
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    Tensor = "Tensor"  # type: ignore

from ..components.ssa import SSAConfig, SelectiveSelfAttention
from ..components.topology import TopologicalSignature, summarise_points


@dataclass
class TLSPolicyConfig:
    embed_dim: int
    num_heads: int = 4
    topk_ratio: float = 0.5
    topo_resolution: int = 16
    topo_sigma: float = 0.1


class TLSPolicy:
    """Small convenience wrapper around the TLS components."""

    def __init__(self, config: TLSPolicyConfig):
        if torch is None:  # pragma: no cover
            raise ImportError("PyTorch is required to build TLSPolicy")
        self.config = config
        ssa_config = SSAConfig(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            topk_ratio=config.topk_ratio,
        )
        self.attention = SelectiveSelfAttention(ssa_config)
        self.last_signature: Optional[TopologicalSignature] = None

    def encode_geometry(self, points: np.ndarray) -> TopologicalSignature:
        """Compute and cache a :class:`TopologicalSignature` from point samples."""

        signature = summarise_points(
            points,
            resolution=(self.config.topo_resolution, self.config.topo_resolution),
            sigma=self.config.topo_sigma,
            build_image=True,
        )
        self.last_signature = signature
        return signature

    def attend(self, hidden_states: Tensor, *, mask: Optional[Tensor] = None) -> Tensor:
        """Forward hidden states through selective self-attention."""

        context, weights = self.attention(hidden_states, mask=mask)
        if self.last_signature is not None and torch.is_grad_enabled():
            grad_norm = float(weights.detach().std().cpu())
            self.attention.update_temperature(grad_norm)
        return context

    def get_temperature(self) -> float:
        return float(self.attention.temperature.detach().cpu())


__all__ = ["TLSPolicy", "TLSPolicyConfig"]
