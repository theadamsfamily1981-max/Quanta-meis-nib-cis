"""Unified Topology-Constrained Free energy (UTCF) loss."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from tff.topo_regularizer import TopologyRegularizer


@dataclass
class UTCFLoss:
    """Combine multiple sources of structure into a scalar training loss."""

    regularizer: TopologyRegularizer
    safety_weight: float = 1.0
    energy_weight: float = 1.0

    def __call__(self, fused_topology, safety_signal, energy) -> Mapping[str, float]:
        fused = np.asarray(fused_topology, dtype=np.float64)
        safety = np.asarray(safety_signal, dtype=np.float64)
        energy = np.asarray(energy, dtype=np.float64)
        if np.shape(fused) != np.shape(safety) or np.shape(fused) != np.shape(energy):
            raise ValueError("Inputs must all share the same shape")

        curvature_ratio = self.regularizer.curvature_ratio(fused)
        safety_penalty = float(np.mean(np.clip(np.multiply(safety, -1.0), a_min=0.0, a_max=None)))
        energy_penalty = float(np.mean(np.clip(energy, a_min=0.0, a_max=None)))

        total = curvature_ratio + self.safety_weight * safety_penalty + self.energy_weight * energy_penalty
        return {
            "total": float(max(total, 0.0)),
            "curvature_ratio": float(curvature_ratio),
            "safety_penalty": float(safety_penalty),
            "energy_penalty": float(energy_penalty),
        }


__all__ = ["UTCFLoss"]
