"""Generate sparse sensor masks based on uncertainty estimates."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SensorMasker:
    """Create deterministic masks using uncertainty scores."""

    threshold: float = 0.5

    def mask(self, scores):
        scores = np.asarray(scores, dtype=np.float64)
        if len(np.shape(scores)) < 2:
            raise ValueError("Expected scores with at least two dimensions")
        mins = np.min(scores, axis=-1, keepdims=True)
        scaled = np.subtract(scores, mins)
        maxs = np.max(scaled, axis=-1, keepdims=True)
        maxs = np.add(maxs, 1e-9)
        scaled = np.true_divide(scaled, maxs)
        return _apply_threshold(scaled, self.threshold)


def _apply_threshold(tensor, threshold):
    if isinstance(tensor, list):
        return [_apply_threshold(value, threshold) for value in tensor]
    return 1.0 if tensor <= threshold else 0.0


__all__ = ["SensorMasker"]
