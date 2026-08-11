"""Lightweight landmark attention for topology aware fusion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class LandmarkAttention:
    """Compute attention scores between queries and landmark descriptors."""

    temperature: float = 1.0

    def attend(self, queries, landmarks) -> Tuple:
        q, l = self._prepare_inputs(queries, landmarks)
        logits = np.matmul(q, np.transpose(l, (0, 2, 1)))
        logits = np.true_divide(logits, max(self.temperature, 1e-6))
        max_logits = np.max(logits, axis=-1, keepdims=True)
        logits = np.subtract(logits, max_logits)
        weights = np.exp(logits)
        normaliser = np.sum(weights, axis=-1, keepdims=True)
        weights = np.true_divide(weights, np.add(normaliser, 1e-9))
        summary = _weighted_sum(weights, l)
        return weights, summary

    def _prepare_inputs(self, queries, landmarks) -> Tuple:
        q = np.asarray(queries, dtype=np.float64)
        l = np.asarray(landmarks, dtype=np.float64)
        if len(np.shape(q)) != 3 or len(np.shape(l)) != 3:
            raise ValueError("Expected 3D tensors for queries and landmarks")
        if np.shape(q)[0] != np.shape(l)[0] or np.shape(q)[2] != np.shape(l)[2]:
            raise ValueError("Batch and feature dimensions must match")
        return q, l


def _weighted_sum(weights, landmarks):
    result = []
    for batch_weights, batch_landmarks in zip(weights, landmarks):
        batch_result = []
        for query_weights in batch_weights:
            vector = []
            for dim in range(len(batch_landmarks[0])):
                total = 0.0
                for weight, descriptor in zip(query_weights, batch_landmarks):
                    total += weight * descriptor[dim]
                vector.append(total)
            batch_result.append(vector)
        result.append(batch_result)
    return result


__all__ = ["LandmarkAttention"]
