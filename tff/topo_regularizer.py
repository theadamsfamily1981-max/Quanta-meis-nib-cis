"""Topology regularisation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class TopologyRegularizer:
    """Analyse fused topology tensors and derive coarse metrics."""

    eps: float = 1e-9

    def curvature_ratio(self, fused: np.ndarray) -> float:
        """Return the ratio between the maximum and mean curvature signal."""

        tensor = np.asarray(fused, dtype=np.float64)
        positive = np.clip(tensor, a_min=0.0, a_max=None)
        maximum = float(np.max(positive))
        mean = float(np.mean(positive) + self.eps)
        return maximum / mean

    def stability_summary(self, fused: np.ndarray) -> Dict[str, float]:
        """Compute stability metrics for the fused tensor."""

        tensor = np.asarray(fused, dtype=np.float64)
        energy = np.square(tensor)
        return {
            "entropy": float(np.mean(-tensor * np.log(tensor + self.eps))),
            "sparsity": float(np.mean(energy < self.eps)),
            "curvature_ratio": self.curvature_ratio(tensor),
        }


__all__ = ["TopologyRegularizer"]
