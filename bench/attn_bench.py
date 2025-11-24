"""Command line helpers to benchmark the attention utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import random
import time

from tfan.tls import TopologicalLandmarkSelector
from tfan.ssa import selective_self_attention


@dataclass
class BenchmarkResult:
    k: int
    duration_s: float
    metric_evaluations: int


def run_landmark_sweep(points: Sequence[Sequence[float]], ks: Sequence[int]) -> Sequence[BenchmarkResult]:
    selector = TopologicalLandmarkSelector()
    results = []
    for k in ks:
        start = time.perf_counter()
        result = selector.select(points, k=k, return_result=True)
        duration = time.perf_counter() - start
        results.append(BenchmarkResult(k, duration, result.metric_evaluations))
    return results


def run_attention_benchmark(size: int, *, seed: int = 0) -> float:
    random.seed(seed)
    scores = [[random.random() for _ in range(size)] for _ in range(size)]
    start = time.perf_counter()
    selective_self_attention(scores)
    return time.perf_counter() - start


__all__ = ["BenchmarkResult", "run_landmark_sweep", "run_attention_benchmark"]


if __name__ == "__main__":
    demo_points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    sweep = run_landmark_sweep(demo_points, ks=[1, 2, 3])
    for result in sweep:
        print(f"k={result.k} evals={result.metric_evaluations} time={result.duration_s:.6f}s")
    print(f"attention benchmark: {run_attention_benchmark(16):.6f}s")
