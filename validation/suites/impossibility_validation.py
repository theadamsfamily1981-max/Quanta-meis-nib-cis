"""Impossibility validation suite with pragmatic theoretical proxies.

The metrics emitted by this script emulate measurements for hard-to-validate
concepts such as halting behavior or Gödel-style incompleteness by using
computational experiments on synthetic data. The results are deterministic and
stored as JSON via the ``--out`` parameter.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Dict, Iterable, List, Tuple


@dataclass
class ImpossibilityMetrics:
    """Metrics capturing proxy values for otherwise intractable concepts."""

    halting_proxy: float
    godel_incompleteness_proxy: float
    p_vs_np_gap: float
    no_free_lunch_proxy: float
    attention_convergence: float
    entropy_resilience: float

    def as_serializable(self) -> Dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def _generate_program_iterations(rng: Random, count: int) -> List[int]:
    return [int(abs(rng.gauss(0.0, 1.0)) * 40) + 1 for _ in range(count)]


def _logistic_map_sequence(rng: Random, size: int, r: float) -> List[float]:
    x = rng.random()
    sequence = []
    for _ in range(size):
        x = r * x * (1.0 - x)
        sequence.append(x)
    return sequence


def _halting_proxy(iterations: Iterable[int]) -> float:
    values = list(iterations)
    clipped = [min(value, 500) for value in values]
    mean_iterations = statistics.mean(clipped)
    return 1.0 / (1.0 + mean_iterations)


def _godel_proxy(sequences: Iterable[List[float]]) -> float:
    symmetry_scores = []
    for sequence in sequences:
        mirrored = list(reversed(sequence))
        correlation = _pearson(sequence, mirrored)
        symmetry_scores.append(abs(correlation))
    return statistics.mean(symmetry_scores)


def _pearson(xs: Iterable[float], ys: Iterable[float]) -> float:
    x_values = list(xs)
    y_values = list(ys)
    if len(x_values) != len(y_values) or not x_values:
        return 0.0
    x_mean = statistics.mean(x_values)
    y_mean = statistics.mean(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in x_values) * sum((y - y_mean) ** 2 for y in y_values)
    )
    if math.isclose(denominator, 0.0):
        return 0.0
    return numerator / denominator


def _p_vs_np_gap(instances: Iterable[Tuple[float, float]]) -> float:
    gaps = [abs(optimal - heuristic) for optimal, heuristic in instances]
    return statistics.mean(gaps)


def _tour_estimates(rng: Random, cities: int) -> Tuple[float, float]:
    distances = [[0.0 for _ in range(cities)] for _ in range(cities)]
    for i in range(cities):
        for j in range(i + 1, cities):
            distance = rng.uniform(1.0, 10.0)
            distances[i][j] = distance
            distances[j][i] = distance
    heuristic = sum(distances[i][(i + 1) % cities] for i in range(cities))
    optimal = _nearest_neighbor(distances)
    return optimal, heuristic


def _nearest_neighbor(distances: List[List[float]]) -> float:
    cities = len(distances)
    unvisited = set(range(1, cities))
    current = 0
    tour_length = 0.0
    while unvisited:
        next_city = min(unvisited, key=lambda city: distances[current][city])
        tour_length += distances[current][next_city]
        current = next_city
        unvisited.remove(next_city)
    tour_length += distances[current][0]
    return tour_length


def _no_free_lunch_proxy(rng: Random, sample_size: int) -> float:
    algorithms = [lambda value: math.sin(value), lambda value: math.cos(value)]
    performances = []
    for algorithm in algorithms:
        results = [abs(algorithm(rng.uniform(-math.pi, math.pi))) for _ in range(sample_size)]
        performances.append(statistics.mean(results))
    difference = abs(performances[0] - performances[1])
    return 1.0 - min(1.0, difference)


def _attention_convergence(sequences: Iterable[List[float]]) -> float:
    convergence = []
    for sequence in sequences:
        peaks = [value for value in sequence if value > 0.75]
        if not sequence:
            continue
        convergence.append(len(peaks) / len(sequence))
    if not convergence:
        return 0.0
    return statistics.mean(convergence)


def _entropy_resilience(sequence_sets: Iterable[List[float]]) -> float:
    entropies = []
    for sequence in sequence_sets:
        histogram = [0 for _ in range(10)]
        for value in sequence:
            index = min(int(value * 10), 9)
            histogram[index] += 1
        total = sum(histogram)
        entropy = 0.0
        for count in histogram:
            if count == 0:
                continue
            probability = count / total
            entropy -= probability * math.log(probability, 2)
        entropies.append(entropy)
    if not entropies:
        return 0.0
    baseline = statistics.mean(entropies)
    variance = statistics.pvariance(entropies)
    resilience = baseline / (1.0 + variance)
    return resilience


def run_suite(seed: int) -> ImpossibilityMetrics:
    rng = Random(seed)
    iteration_counts = _generate_program_iterations(rng, 256)
    logistic_sequences = [_logistic_map_sequence(rng, 128, r) for r in (3.7, 3.85, 3.95)]
    tour_instances = [_tour_estimates(rng, 6) for _ in range(20)]

    metrics = ImpossibilityMetrics(
        halting_proxy=_halting_proxy(iteration_counts),
        godel_incompleteness_proxy=_godel_proxy(logistic_sequences),
        p_vs_np_gap=_p_vs_np_gap(tour_instances),
        no_free_lunch_proxy=_no_free_lunch_proxy(rng, 400),
        attention_convergence=_attention_convergence(logistic_sequences),
        entropy_resilience=_entropy_resilience(logistic_sequences),
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the impossibility validation suite.")
    parser.add_argument("--out", type=Path, required=True, help="Path to the JSON file that will store the metrics.")
    parser.add_argument(
        "--seed",
        type=int,
        default=4242,
        help="Seed for the pseudo-random generator to ensure reproducibility (default: 4242).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_suite(seed=args.seed)
    payload = {
        "metrics": metrics.as_serializable(),
        "metadata": {
            "seed": args.seed,
            "logistic_parameters": [3.7, 3.85, 3.95],
            "tour_cities": 6,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
