"""Core routines for GR-TES Phase III refinement."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

Vector = List[float]
PointCloud = List[Vector]


def _clone(points: Sequence[Sequence[float]]) -> PointCloud:
    return [list(p) for p in points]


def _finite_difference_gradient(
    potential_fn: Callable[[PointCloud], float],
    position: PointCloud,
    epsilon: float = 1e-4,
) -> PointCloud:
    grad = [[0.0 for _ in vec] for vec in position]
    for i, vec in enumerate(position):
        for j in range(len(vec)):
            forward = _clone(position)
            backward = _clone(position)
            forward[i][j] += epsilon
            backward[i][j] -= epsilon
            grad[i][j] = (
                potential_fn(forward) - potential_fn(backward)
            ) / (2.0 * epsilon)
    return grad


@dataclass
class LangevinConfig:
    step_size: float = 1e-2
    temperature: float = 1.0
    n_steps: int = 100
    gradient_fn: Optional[Callable[[PointCloud], PointCloud]] = None


def langevin_dynamics(
    initial_positions: Sequence[Sequence[float]],
    potential_fn: Callable[[PointCloud], float],
    config: LangevinConfig,
    rng: Optional[random.Random] = None,
) -> PointCloud:
    rng = random.Random() if rng is None else rng
    positions = _clone(initial_positions)

    beta = 1.0 / max(config.temperature, 1e-12)
    sqrt_term = math.sqrt(2.0 / beta * config.step_size)

    for _ in range(config.n_steps):
        if config.gradient_fn is None:
            gradient = _finite_difference_gradient(potential_fn, positions)
        else:
            gradient = config.gradient_fn(positions)

        for i, vec in enumerate(positions):
            for j in range(len(vec)):
                noise = rng.gauss(0.0, 1.0)
                vec[j] -= config.step_size * gradient[i][j]
                vec[j] += sqrt_term * noise

    return positions


def _pairwise_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def gaussian_smooth(points: Sequence[Sequence[float]], bandwidth: float) -> PointCloud:
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    points = _clone(points)
    n = len(points)
    if n == 0:
        return []

    weights = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist = _pairwise_distance(points[i], points[j])
            weights[i][j] = math.exp(-0.5 * (dist ** 2) / (bandwidth ** 2))
        row_sum = sum(weights[i])
        if row_sum == 0:
            continue
        for j in range(n):
            weights[i][j] /= row_sum

    smoothed: PointCloud = [[0.0 for _ in points[0]] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k, value in enumerate(points[j]):
                smoothed[i][k] += weights[i][j] * value
    return smoothed


@dataclass
class WitnessComplex:
    landmarks: PointCloud
    witnesses: PointCloud
    k: int = 5

    def pairwise_distances(
        self, a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]
    ) -> List[List[float]]:
        return [[_pairwise_distance(pa, pb) for pb in b] for pa in a]

    def build(self) -> dict[str, List[tuple[int, ...]]]:
        distances = self.pairwise_distances(self.witnesses, self.landmarks)
        nearest = [
            sorted(range(len(row)), key=lambda idx: row[idx])[: self.k]
            for row in distances
        ]

        vertices = list(range(len(self.landmarks)))
        edges: set[tuple[int, int]] = set()
        for row in nearest:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    edge = tuple(sorted((row[i], row[j])))
                    edges.add(edge)

        return {"vertices": [(v,) for v in vertices], "edges": sorted(edges)}


__all__ = [
    "LangevinConfig",
    "gaussian_smooth",
    "langevin_dynamics",
    "WitnessComplex",
]
