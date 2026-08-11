"""Persistent homology helpers used to monitor topology of learning dynamics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

try:  # pragma: no cover - optional dependency
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - allow skipping functionality when numpy absent
    np = None  # type: ignore[assignment]

try:  # pragma: no cover - optional import for tests environments without ripser
    from ripser import ripser
except Exception:  # pragma: no cover - fallback when ripser is unavailable
    ripser = None  # type: ignore


@dataclass
class PersistenceResult:
    """Container for persistence diagrams and auxiliary statistics."""

    diagrams: Tuple[np.ndarray, ...]
    bottleneck: float


class PersistentHomologyAnalyser:
    """Compute persistent homology and bottleneck stability estimates."""

    def __init__(self, maxdim: int = 1) -> None:
        self.maxdim = maxdim

    def compute(self, points: Iterable[Iterable[float]]) -> PersistenceResult:
        """Compute diagrams for a point cloud and derive a bottleneck bound."""

        if ripser is None:  # pragma: no cover - executed only when dependency missing
            raise RuntimeError(
                "ripser is not available. Install the optional dependency to compute persistent homology."
            )
        if np is None:
            raise RuntimeError("numpy is required to compute persistent homology")
        point_array = np.asarray(tuple(points), dtype=float)
        if point_array.ndim != 2:
            raise ValueError("Point cloud must be a 2-D array")
        result = ripser(point_array, maxdim=self.maxdim)
        diagrams = tuple(result["dgms"])
        bottleneck = 0.0
        for diagram in diagrams:
            if len(diagram) == 0:
                continue
            lifetimes = diagram[:, 1] - diagram[:, 0]
            bottleneck = max(bottleneck, float(np.max(lifetimes)))
        return PersistenceResult(diagrams=diagrams, bottleneck=bottleneck)


__all__: Tuple[str, ...] = ("PersistenceResult", "PersistentHomologyAnalyser")
