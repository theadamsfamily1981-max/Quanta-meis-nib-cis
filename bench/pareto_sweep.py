"""Objective evaluation helper."""
from __future__ import annotations

from typing import Sequence

from tfan.multiobj import expected_hypervolume_improvement, pareto_front


def sweep(objectives: Sequence[Sequence[float]], candidates: Sequence[Sequence[float]], reference: Sequence[float]) -> Sequence[float]:
    front = pareto_front(objectives)
    return expected_hypervolume_improvement(front, candidates, reference)


__all__ = ["sweep"]


if __name__ == "__main__":
    objectives = [[0.2, 0.9], [0.7, 0.3], [0.5, 0.5]]
    candidates = [[0.6, 0.6], [0.3, 0.8]]
    reference = [0.0, 0.0]
    print(sweep(objectives, candidates, reference))
