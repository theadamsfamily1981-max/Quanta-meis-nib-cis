"""Metric utility scaffolding."""

from __future__ import annotations

from typing import Any, Dict, Iterable


def compute(metrics: Iterable[str], predictions: Any, targets: Any) -> Dict[str, float]:
    """Compute placeholder metrics for the framework."""

    _ = (metrics, predictions, targets)
    return {}
