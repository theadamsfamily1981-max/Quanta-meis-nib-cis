"""Demonstration script for the Phase III refinement pipeline."""
from __future__ import annotations

import argparse
import pathlib
import random
import sys
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from core.grtes_core import LangevinConfig, langevin_dynamics
from core.multi_scale_PH import MultiScalePHHierarchy


def double_well_potential(points: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for vec in points:
        for value in vec:
            total += 0.25 * (value ** 2 - 1) ** 2
    return total


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-points", type=int, default=64, help="Number of samples")
    parser.add_argument(
        "--steps", type=int, default=256, help="Number of Langevin integration steps"
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="*",
        default=(0.1, 0.25, 0.5),
        help="Gaussian smoothing scales",
    )
    args = parser.parse_args(argv)

    rng = random.Random(0)
    initial = [[rng.gauss(0.0, 1.5) for _ in range(2)] for _ in range(args.n_points)]
    config = LangevinConfig(step_size=1e-2, temperature=0.5, n_steps=args.steps)
    samples = langevin_dynamics(initial, double_well_potential, config, rng=rng)

    hierarchy = MultiScalePHHierarchy(k_landmarks=16, witness_neighbors=6, seed=0)
    results = hierarchy.build_hierarchy(samples, scales=args.scales)

    print("Generated hierarchy with", len(results), "scales")
    for result in results:
        persistence_summary = [pair.persistence for pair in result.diagram]
        average = sum(persistence_summary) / len(persistence_summary) if persistence_summary else 0.0
        print(
            f"Scale={result.scale:.3f} | #simplices={len(result.complex_summary['edges'])}"
            f" | Avg persistence={average:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
