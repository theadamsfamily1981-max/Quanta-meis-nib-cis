"""Command line interface for TFAN plotting demos."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from .pipeline import run_pipeline


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TFAN pipeline demo")
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--token-dim",
        type=int,
        default=8,
        help="Dimensionality of the sampled token embeddings.",
    )
    parser.add_argument(
        "--save-plots",
        type=Path,
        default=None,
        help="Directory to save diagnostic plots. If omitted plots are not persisted.",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit K sweep for the throughput diagnostic.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    losses = run_pipeline(
        seed=args.seed,
        token_dim=args.token_dim,
        k_values=args.k_values,
        save_plots=args.save_plots,
    )

    metrics = losses.as_dict()
    longest = max(len(name) for name in metrics)
    for name, value in metrics.items():
        print(f"{name.rjust(longest)} : {value:.4f}")
    if args.save_plots is not None:
        print(f"Plots saved to: {Path(args.save_plots).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
