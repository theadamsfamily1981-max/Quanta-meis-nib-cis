"""Safety monitoring utilities for T-FAN."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass
class SafetyMonitor:
    """Compute true/false positive safety rates."""

    threshold: float = 0.0

    def evaluate(self, logits, labels) -> Mapping[str, float]:
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        if np.shape(logits) != np.shape(labels):
            raise ValueError("Logits and labels must share the same shape")

        predictions = [value >= self.threshold for value in _flatten(logits)]
        positives = [value > 0.5 for value in _flatten(labels)]

        tp = sum(1.0 for pred, pos in zip(predictions, positives) if pred and pos)
        fp = sum(1.0 for pred, pos in zip(predictions, positives) if pred and not pos)
        tn = sum(1.0 for pred, pos in zip(predictions, positives) if not pred and not pos)
        fn = sum(1.0 for pred, pos in zip(predictions, positives) if not pred and pos)

        tp_rate = tp / max(tp + fn, 1.0)
        fp_rate = fp / max(fp + tn, 1.0)
        return {
            "TP": tp_rate,
            "FP": fp_rate,
        }


def _flatten(tensor):
    if isinstance(tensor, list):
        for value in tensor:
            yield from _flatten(value)
    else:
        yield float(tensor)


__all__ = ["SafetyMonitor"]
