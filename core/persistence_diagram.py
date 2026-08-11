"""Light-weight persistence diagram utilities relying only on the stdlib."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

Point = Sequence[float]


@dataclass
class PersistencePair:
    birth: float
    death: float

    @property
    def persistence(self) -> float:
        return self.death - self.birth


def _pairwise_distance(a: Point, b: Point) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def zero_dimensional_persistence(points: Sequence[Point]) -> List[PersistencePair]:
    points = [tuple(p) for p in points]
    n = len(points)
    if n == 0:
        return []

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((_pairwise_distance(points[i], points[j]), i, j))
    edges.sort()

    parent = list(range(n))
    size = [1] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> bool:
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if size[rx] < size[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        size[rx] += size[ry]
        return True

    diagram: List[PersistencePair] = []
    for weight, i, j in edges:
        if union(i, j):
            diagram.append(PersistencePair(birth=0.0, death=weight))

    return diagram


__all__ = ["PersistencePair", "zero_dimensional_persistence"]
