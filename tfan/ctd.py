"""Curvature tuned deployment gate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def _validate(values: Sequence[float]) -> Sequence[float]:
    if not values:
        raise ValueError("values must not be empty")
    return tuple(float(v) for v in values)


@dataclass
class CurvatureGate:
    """Smoothing-based deployment gate.

    The gate consumes a sequence of curvature estimates (or any quantity that
    behaves similarly) and computes a smoothed trend.  The :meth:`allow` method
    returns ``True`` when the smoothed trend stays below ``max_curvature``.
    """

    smoothing: float = 0.5
    max_curvature: float = 0.1

    def __post_init__(self) -> None:
        if not 0 < self.smoothing <= 1:
            raise ValueError("smoothing must lie in (0, 1]")
        if self.max_curvature <= 0:
            raise ValueError("max_curvature must be positive")

    def smooth(self, values: Sequence[float]) -> float:
        values = _validate(values)
        estimate = values[0]
        for value in values[1:]:
            estimate = self.smoothing * value + (1 - self.smoothing) * estimate
        return estimate

    def allow(self, values: Sequence[float]) -> bool:
        return self.smooth(values) <= self.max_curvature


__all__ = ["CurvatureGate"]
