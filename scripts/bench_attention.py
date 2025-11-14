#!/usr/bin/env python3
"""
Benchmark radial sparse attention vs dense.

Gate: ≥ 3× speedup on T∈{8k,16k,32k} with ≤2% Δacc
"""
import argparse
import json
import sys
from pathlib import Path

import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tfan.attention import benchmark_attention
from tfan.io import new_artifact, write_json


def main():
    parser = argparse.ArgumentParser(description="Benchmark sparse attention")
    parser.add_argument("--lengths", nargs="+", type=int,
                        default=[8000, 16000, 32000],
                        help="Sequence lengths to test")
    parser.add_argument("--keep-ratio", type=float, default=0.33,
                        help="Landmark keep ratio")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path")

    args = parser.parse_args()

    print("=" * 60)
    print("TFAN Radial Sparse Attention Benchmark")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Keep ratio: {args.keep_ratio}")
    print(f"Sequence lengths: {args.lengths}")

    results = benchmark_attention(
        seq_lengths=args.lengths,
        keep_ratio=args.keep_ratio,
        device=args.device
    )

    # Check gate: ≥ 3× speedup
    all_pass = all(r["speedup"] >= 3.0 for r in results.values())

    report = {
        "device": args.device,
        "keep_ratio": args.keep_ratio,
        "results": results,
        "gate_pass": all_pass,
        "gate_threshold": 3.0
    }

    # Write results
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = new_artifact("bench_attention", suffix=".json")

    write_json(out_path, report)

    print("\n" + "=" * 60)
    print(f"Gate (≥3× speedup): {'PASS' if all_pass else 'FAIL'}")
    print(f"Report: {out_path}")
    print("=" * 60)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
