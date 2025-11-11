"""Topology utilities used by TLS policies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import numpy as np

ArrayLike = np.ndarray


@dataclass
class TopologicalSignature:
    """Holds persistent homology style statistics."""

    lifetimes: ArrayLike
    persistence_image: Optional[ArrayLike]
    total_weight: float

    def to_dict(self) -> dict:
        result = {
            "lifetimes": self.lifetimes.tolist(),
            "total_weight": float(self.total_weight),
        }
        if self.persistence_image is not None:
            result["persistence_image"] = self.persistence_image.tolist()
        return result


def pairwise_distances(points: ArrayLike) -> ArrayLike:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("`points` must be a 2D array")
    squared = np.sum(points**2, axis=1, keepdims=True)
    distances = squared - 2 * points @ points.T + squared.T
    np.maximum(distances, 0.0, out=distances)
    return np.sqrt(distances, out=distances)


def minimum_spanning_tree(distances: ArrayLike) -> Tuple[ArrayLike, float]:
    distances = np.asarray(distances, dtype=np.float64)
    n = distances.shape[0]
    if distances.shape != (n, n):
        raise ValueError("`distances` must be a square matrix")

    selected = np.zeros(n, dtype=bool)
    selected[0] = True
    mst = np.zeros_like(distances)
    total_weight = 0.0

    for _ in range(n - 1):
        mask = np.outer(selected, ~selected)
        candidate_weights = np.where(mask, distances, np.inf)
        idx = np.argmin(candidate_weights)
        i, j = divmod(idx, n)
        weight = candidate_weights[i, j]
        if not np.isfinite(weight):
            raise RuntimeError("Input graph appears disconnected")
        mst[i, j] = mst[j, i] = weight
        total_weight += float(weight)
        selected[j] = True

    return mst, total_weight


def persistence_lifetimes(mst: ArrayLike) -> ArrayLike:
    edges = mst[np.triu_indices_from(mst, k=1)]
    edges = edges[edges > 0]
    if edges.size == 0:
        return np.zeros(1, dtype=np.float64)
    lifetimes = np.sort(edges)
    gaps = np.diff(np.concatenate([[0.0], lifetimes]))
    gaps[::-1].sort()
    return gaps


def build_persistence_image(
    lifetimes: ArrayLike,
    resolution: Tuple[int, int] = (16, 16),
    sigma: float = 0.1,
    value_range: Optional[Tuple[float, float]] = None,
) -> ArrayLike:
    lifetimes = np.asarray(lifetimes, dtype=np.float64)
    if lifetimes.ndim != 1:
        raise ValueError("`lifetimes` must be a 1D array")
    if lifetimes.size == 0:
        return np.zeros(resolution, dtype=np.float64)

    min_v, max_v = value_range or (float(lifetimes.min()), float(lifetimes.max()))
    if math.isclose(min_v, max_v):
        max_v += 1.0
    grid_x = np.linspace(min_v, max_v, resolution[0])
    grid_y = np.linspace(min_v, max_v, resolution[1])
    xv, yv = np.meshgrid(grid_x, grid_y, indexing="ij")

    image = np.zeros_like(xv)
    for value in lifetimes:
        image += np.exp(-((xv - value) ** 2 + (yv - value) ** 2) / (2 * sigma**2))
    image /= lifetimes.size
    return image


def summarise_points(
    points: ArrayLike,
    *,
    resolution: Tuple[int, int] = (16, 16),
    sigma: float = 0.1,
    build_image: bool = True,
) -> TopologicalSignature:
    distances = pairwise_distances(points)
    mst, total_weight = minimum_spanning_tree(distances)
    lifetimes = persistence_lifetimes(mst)
    image = build_persistence_image(lifetimes, resolution=resolution, sigma=sigma) if build_image else None
    return TopologicalSignature(lifetimes=lifetimes, persistence_image=image, total_weight=total_weight)


__all__ = [
    "TopologicalSignature",
    "pairwise_distances",
    "minimum_spanning_tree",
    "persistence_lifetimes",
    "build_persistence_image",
    "summarise_points",
]
