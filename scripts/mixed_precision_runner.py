"""Utilities for running mixed precision training experiments."""
from __future__ import annotations

import argparse

def build_argparser() -> argparse.ArgumentParser:
    """Create the argument parser for the mixed precision runner."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--lr', type=float, default=0.2,
                    help='Learning rate for the optimizer (default: 0.2).')
    ap.add_argument('--rho-safe', type=float, default=0.8,
                    help='Safety factor for rho (default: 0.8).')
    ap.add_argument('--tau', type=float, default=1.0,
                    help='Tau parameter for stabilization (default: 1.0).')
    ap.add_argument('--warmup', type=int, default=200,
                    help='Number of warmup steps (default: 200).')
    return ap

def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    return build_argparser().parse_args(args)

def main() -> None:
    """Entry point for command line usage."""
    parse_args()

if __name__ == '__main__':
    main()
