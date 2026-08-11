"""Energy balancing helper routines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass
class EnergyBalancer:
    """Track free-energy budgets and compute residuals."""

    baseline: float = 1.0

    def measure(self, signal: np.ndarray) -> Mapping[str, float]:
        tensor = np.asarray(signal, dtype=np.float64)
        positive = np.clip(tensor, a_min=0.0, a_max=None)
        mean_energy = float(np.mean(positive))
        reduction = 100.0 * (1.0 - mean_energy / max(self.baseline, 1e-6))
        return {
            "mean_energy": mean_energy,
            "lora_reduction": reduction,
        }


__all__ = ["EnergyBalancer"]
