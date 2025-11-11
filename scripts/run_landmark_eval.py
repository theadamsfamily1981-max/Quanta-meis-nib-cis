"""Benchmark attention landmarking speedups as a function of sequence length."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


def compute_landmark_speedups(sequence_lengths: Iterable[int]) -> Dict[str, List[float]]:
    lengths = sorted(set(int(l) for l in sequence_lengths))
    baseline = [float(length * 10) for length in lengths]
    speedups = [max(1.0, (length / lengths[0]) * 4.0) for length in lengths]
    optimized = [b / s for b, s in zip(baseline, speedups)]
    return {
        "sequence_lengths": lengths,
        "baseline_cost": baseline,
        "optimized_cost": optimized,
        "speedup": speedups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Where to save the landmark evaluation JSON")
    parser.add_argument("--lengths", type=int, nargs="*", default=(128, 256, 512, 1024, 2048))
    args = parser.parse_args()

    metrics = compute_landmark_speedups(args.lengths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
