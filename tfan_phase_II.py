"""TFAN Phase II orchestration helpers."""

from __future__ import annotations

from typing import Iterable

from tfan_phase_I import summarize_inputs


def normalize_series(series: Iterable[float]) -> list[float]:
    """Normalize ``series`` to a 0-1 range using min-max scaling."""

    values = list(series)
    if not values:
        return []

    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        return [0.0 for _ in values]

    span = maximum - minimum
    return [(value - minimum) / span for value in values]


def phase_two_projection(series: Iterable[float]) -> float:
    """Project the normalized series back into the TFAN average domain."""

    normalized = normalize_series(series)
    return summarize_inputs(normalized)
