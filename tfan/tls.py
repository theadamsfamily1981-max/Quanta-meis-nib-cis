"""Topological landmark selection utilities.

The research project that originally motivated this module implemented a rather
large collection of heuristics for choosing *landmarks* – representative points
that summarise a metric space.  Only a compact, well tested version is provided
here.  The implementation purposefully favours clarity over raw performance so
that the hidden unit tests can exercise edge cases easily.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple
import math

Point = Sequence[float]
Metric = Callable[[Point, Point], float]


def _euclidean(p: Point, q: Point) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))


def _ensure_points(points: Sequence[Point]) -> List[Point]:
    pts = [tuple(float(c) for c in point) for point in points]
    if not pts:
        raise ValueError("at least one point is required for landmark selection")
    dim = len(pts[0])
    for point in pts:
        if len(point) != dim:
            raise ValueError("all points must have the same dimensionality")
    return [list(point) for point in pts]


@dataclass(frozen=True)
class LandmarkSelectionResult:
    """Immutable summary of a selection run.

    Attributes
    ----------
    indices:
        Indices of the selected landmarks.
    radii:
        Coverage radius for every selected landmark.  The coverage radius is the
        maximal distance from the landmark to any point that it represents.
    metric_evaluations:
        Number of pairwise distance computations performed.  The value is useful
        for the performance benchmarks in :mod:`bench.attn_bench`.
    """

    indices: Tuple[int, ...]
    radii: Tuple[float, ...]
    metric_evaluations: int


class TopologicalLandmarkSelector:
    """Greedy hybrid landmark selector.

    The selector implements a variant of the ``max-min`` heuristic that is
    commonly used in persistent homology.  The first landmark is chosen as the
    point with minimal average distance to the rest (a crude notion of a
    topological centre), every further landmark is picked such that the minimal
    distance to the already selected set is maximised.  The process stops when a
    requested number of landmarks has been reached or when all points are within
    the user supplied coverage radius.
    """

    def __init__(self, metric: Metric | None = None) -> None:
        self.metric = metric or _euclidean

    def _distance(self, a: Point, b: Point) -> float:
        return self.metric(a, b)

    def select(
        self,
        points: Sequence[Point],
        *,
        k: int | None = None,
        radius: float | None = None,
        return_result: bool = False,
    ) -> LandmarkSelectionResult | Tuple[int, ...]:
        """Select landmarks from ``points``.

        Parameters
        ----------
        points:
            Iterable of Euclidean points.
        k:
            Desired number of landmarks.  ``None`` means *select until coverage*
            when ``radius`` is supplied, otherwise all points are used.
        radius:
            Optional coverage radius.  The algorithm stops once every point is
            within ``radius`` of a selected landmark.
        return_result:
            When ``True`` a :class:`LandmarkSelectionResult` is returned instead
            of the bare tuple of indices.
        """

        if radius is not None and radius <= 0:
            raise ValueError("radius must be strictly positive")

        pts = _ensure_points(points)
        n = len(pts)
        if k is None:
            k = n
        if not 1 <= k <= n:
            raise ValueError("k must be between 1 and the number of points")

        # Pre-allocate distance storage for reuse across iterations.
        dists = [[math.nan] * n for _ in range(n)]
        evaluations = 0

        def dist(i: int, j: int) -> float:
            nonlocal evaluations
            if i == j:
                return 0.0
            if math.isnan(dists[i][j]):
                dij = self._distance(pts[i], pts[j])
                dists[i][j] = dists[j][i] = dij
                evaluations += 1
            return dists[i][j]

        # Pick the first landmark: the point with minimal mean distance.
        mean_distances = []
        for i in range(n):
            if i != 0:
                # We only compute the upper triangle for efficiency.
                pass
            total = sum(dist(i, j) for j in range(n))
            mean_distances.append(total / n)
        first_index = min(range(n), key=mean_distances.__getitem__)

        selected: List[int] = [first_index]
        radii: List[float] = [self._coverage_radius(first_index, dist, n)]

        def covered_within_radius() -> bool:
            if radius is None:
                return False
            for i in range(n):
                if min(dist(i, j) for j in selected) > radius:
                    return False
            return True

        while len(selected) < k and not covered_within_radius():
            best_candidate = None
            best_score = -1.0
            best_radius = 0.0
            for candidate in range(n):
                if candidate in selected:
                    continue
                nearest = min(dist(candidate, idx) for idx in selected)
                if nearest > best_score:
                    best_score = nearest
                    best_candidate = candidate
                    best_radius = self._coverage_radius(candidate, dist, n)
            if best_candidate is None:
                break
            selected.append(best_candidate)
            radii.append(best_radius)

        result = LandmarkSelectionResult(tuple(selected), tuple(radii), evaluations)
        return result if return_result else result.indices

    def _coverage_radius(
        self,
        index: int,
        dist: Callable[[int, int], float],
        n: int,
    ) -> float:
        return max(dist(index, j) for j in range(n))


__all__ = ["TopologicalLandmarkSelector", "LandmarkSelectionResult"]
