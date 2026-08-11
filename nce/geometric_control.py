"""Simple geometric controller for normalised energies."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GeometricController:
    """Balance energy between manifolds using a convex update."""

    momentum: float = 0.2

    def step(self, state, target):
        state = np.asarray(state, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if np.shape(state) != np.shape(target):
            raise ValueError("State and target must share the same shape")
        delta = np.subtract(target, state)
        update = np.multiply(delta, self.momentum)
        updated = np.add(state, update)
        updated = np.clip(updated, a_min=0.0, a_max=None)
        return updated


__all__ = ["GeometricController"]
