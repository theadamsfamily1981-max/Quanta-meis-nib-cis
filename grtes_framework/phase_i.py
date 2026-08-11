"""Phase I primitives for the GRTES framework."""

from __future__ import annotations

from statistics import mean
from typing import Iterable


def compute_baseline(stream: Iterable[float]) -> float:
    """Compute a baseline GRTES signal from the provided stream."""

    values = list(stream)
    if not values:
        raise ValueError("stream must contain at least one value")
    return mean(values)
