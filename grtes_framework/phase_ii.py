"""Phase II coordination logic for the GRTES framework."""

from __future__ import annotations

from typing import Iterable

from .phase_i import compute_baseline


def compute_adjusted_baseline(stream: Iterable[float], *, boost: float = 0.1) -> float:
    """Boost the baseline using the provided ``boost`` factor."""

    baseline = compute_baseline(stream)
    return baseline * (1 + boost)
