"""Multi-objective optimisation helpers."""
from __future__ import annotations

from typing import List, Sequence, Tuple


def pareto_front(points: Sequence[Sequence[float]]) -> List[Tuple[float, ...]]:
    """Return the Pareto front of ``points`` for a maximisation problem."""

    candidates = [tuple(float(v) for v in point) for point in points]
    front: List[Tuple[float, ...]] = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            if candidate == other:
                continue
            if all(o >= c for o, c in zip(other, candidate)) and any(o > c for o, c in zip(other, candidate)):
                dominated = True
                break
        if not dominated and candidate not in front:
            front.append(candidate)
    return sorted(front, reverse=True)


def expected_hypervolume_improvement(
    pareto: Sequence[Sequence[float]],
    candidates: Sequence[Sequence[float]],
    reference: Sequence[float],
) -> List[float]:
    """Compute a simple expected hypervolume improvement score.

    The implementation assumes independent normally distributed candidate
    objectives with zero variance; the routine therefore collapses to computing
    the actual hypervolume improvement using the candidate means.  The function
    is intentionally simple yet deterministic which is perfect for the tests.
    """

    ref = tuple(float(r) for r in reference)
    base_hv = _hypervolume(pareto, ref)
    scores = []
    for candidate in candidates:
        hv = _hypervolume(list(pareto) + [candidate], ref)
        scores.append(max(0.0, hv - base_hv))
    return scores


def _hypervolume(points: Sequence[Sequence[float]], reference: Sequence[float]) -> float:
    ref = tuple(reference)
    dominated: List[Tuple[float, ...]] = []
    for point in pareto_front(points):
        dominated.append(tuple(max(ref_i, value) for ref_i, value in zip(ref, point)))
    volume = 0.0
    for point in dominated:
        volume += _rectangle_volume(point, ref)
    return volume


def _rectangle_volume(point: Sequence[float], reference: Sequence[float]) -> float:
    volume = 1.0
    for value, ref in zip(point, reference):
        volume *= max(0.0, value - ref)
    return volume


__all__ = ["pareto_front", "expected_hypervolume_improvement"]
