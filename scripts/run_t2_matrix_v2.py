#!/usr/bin/env python3
"""Launch the T2 retune matrix for the v1.1c ACR-W configuration.

The matrix executes 50 runs by default to validate that the stability metric
exceeds 0.90 on at least 70% of runs while enforcing ||∇L|| < 0.5.

The script builds commands for a generic trainer entrypoint and can be adapted
by modifying ``TRAIN_ENTRYPOINT`` or overriding via the ``--entrypoint`` flag.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments" / "exp_t2_acrw_v1_1c.yaml"
TRAIN_ENTRYPOINT = "python train.py"


def build_command(seed: int, entrypoint: str, config_path: Path) -> List[str]:
    """Construct the execution command for a single seed."""
    return [
        *entrypoint.split(),
        "--config",
        str(config_path),
        "--seed",
        str(seed),
        "--targets",
        "stability>=0.90",
        "grad_norm<=0.5",
    ]


def generate_matrix(seeds: Iterable[int]) -> List[int]:
    """Return the ordered seeds used for the matrix run."""
    return list(seeds)


def run_matrix(
    seeds: Iterable[int],
    entrypoint: str,
    config_path: Path,
    dry_run: bool,
) -> None:
    """Execute or print the commands required for the matrix."""
    commands = [build_command(seed, entrypoint, config_path) for seed in seeds]

    for command in commands:
        if dry_run:
            print("DRY-RUN:", " ".join(command))
            continue

        print("EXECUTING:", " ".join(command))
        subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the T2 v1.1c matrix")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the base experiment configuration.",
    )
    parser.add_argument(
        "--entrypoint",
        default=TRAIN_ENTRYPOINT,
        help="Training command used to launch each experiment.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=50,
        help="Number of runs to execute in the matrix.",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Offset applied to the generated seed sequence.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = generate_matrix(range(args.seed_offset, args.seed_offset + args.runs))
    run_matrix(seeds, args.entrypoint, args.config.resolve(), args.dry_run)


if __name__ == "__main__":
    main()
