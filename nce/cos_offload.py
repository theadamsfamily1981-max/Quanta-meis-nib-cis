"""Co-operative offloading strategy primitives."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CosineOffloader:
    """Distribute workloads according to cosine similarity scores."""

    bias: float = 0.0

    def route(self, anchors, workers):
        anchors, workers = self._prepare_inputs(anchors, workers)
        norm_anchors = self._normalise(anchors)
        norm_workers = self._normalise(workers)
        logits = np.matmul(norm_anchors, np.transpose(norm_workers, (0, 2, 1)))
        logits = np.add(logits, self.bias)
        max_logits = np.max(logits, axis=-1, keepdims=True)
        logits = np.subtract(logits, max_logits)
        weights = np.exp(logits)
        normaliser = np.sum(weights, axis=-1, keepdims=True)
        weights = np.true_divide(weights, np.add(normaliser, 1e-9))
        return weights

    def _prepare_inputs(self, anchors, workers):
        a = np.asarray(anchors, dtype=np.float64)
        w = np.asarray(workers, dtype=np.float64)
        if len(np.shape(a)) != 3 or len(np.shape(w)) != 3:
            raise ValueError("Expected 3D tensors for anchors and workers")
        if np.shape(a)[0] != np.shape(w)[0] or np.shape(a)[2] != np.shape(w)[2]:
            raise ValueError("Mismatched batch or feature sizes")
        return a, w

    def _normalise(self, tensor):
        norms = np.linalg.norm(tensor, axis=-1, keepdims=True)
        norms = np.add(norms, 1e-9)
        return np.true_divide(tensor, norms)


__all__ = ["CosineOffloader"]
