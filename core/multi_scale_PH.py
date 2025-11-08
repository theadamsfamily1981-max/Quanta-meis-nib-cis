"""Multi-scale persistent homology hierarchy utilities."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence

from .grtes_core import WitnessComplex, gaussian_smooth
from .persistence_diagram import PersistencePair, zero_dimensional_persistence

PointCloud = List[List[float]]


@dataclass
class ScaleResult:
    scale: float
    smoothed_points: PointCloud
    diagram: List[PersistencePair]
    complex_summary: dict[str, List[tuple[int, ...]]]


class MultiScalePHHierarchy:
    """Bundle together smoothing, complex construction, and persistence."""

    def __init__(
        self, k_landmarks: int = 16, witness_neighbors: int = 6, seed: int = 0
    ) -> None:
        self.k_landmarks = k_landmarks
        self.witness_neighbors = witness_neighbors
        self._rng = random.Random(seed)

    def _select_landmarks(self, points: PointCloud) -> PointCloud:
        if len(points) <= self.k_landmarks:
            return [list(p) for p in points]
        indices = list(range(len(points)))
        self._rng.shuffle(indices)
        chosen = indices[: self.k_landmarks]
        return [list(points[i]) for i in chosen]

    def build_hierarchy(
        self, points: Sequence[Sequence[float]], scales: Sequence[float]
    ) -> List[ScaleResult]:
        base_points = [list(p) for p in points]
        witnesses = [list(p) for p in base_points]
        results: List[ScaleResult] = []

        for scale in scales:
            smoothed = gaussian_smooth(base_points, bandwidth=scale)
            landmarks = self._select_landmarks(smoothed)
            complex_builder = WitnessComplex(
                landmarks=landmarks, witnesses=witnesses, k=self.witness_neighbors
            )
            complex_summary = complex_builder.build()
            diagram = zero_dimensional_persistence(smoothed)
            results.append(
                ScaleResult(
                    scale=scale,
                    smoothed_points=smoothed,
                    diagram=diagram,
                    complex_summary=complex_summary,
                )
            )
        return results


__all__ = ["MultiScalePHHierarchy", "ScaleResult"]
