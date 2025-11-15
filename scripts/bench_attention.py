#!/usr/bin/env python
"""
Benchmark sparse attention performance.

Usage:
    python scripts/bench_attention.py --seq 8192 16384 32768 --batch 4
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from tfan.attention import SparseAttention, benchmark_attention


def main():
    parser = argparse.ArgumentParser(description="Benchmark sparse attention")
    parser.add_argument("--seq", type=int, nargs="+", default=[8192, 16384, 32768],
                        help="Sequence lengths to test")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--d-model", type=int, default=768, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=12, help="Number of heads")
    parser.add_argument("--n-runs", type=int, default=30, help="Number of runs per length")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--report", type=str, default="artifacts/bench/attention.json",
                        help="Output report path")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup runs")
    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TF-A-N Sparse Attention Benchmark")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch}")
    print(f"Model dim: {args.d_model}")
    print(f"Heads: {args.n_heads}")
    print(f"Sequence lengths: {args.seq}")
    print(f"Runs per length: {args.n_runs}")
    print("-" * 80)

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Create models
    sparse_attn = SparseAttention(
        d_model=args.d_model,
        n_heads=args.n_heads,
        keep_ratio=0.33,
        alpha=0.7,
        window_size=128,
    ).to(args.device)

    full_attn = nn.MultiheadAttention(
        args.d_model,
        args.n_heads,
        batch_first=True,
    ).to(args.device)

    results = {
        "config": {
            "device": args.device,
            "batch_size": args.batch,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_runs": args.n_runs,
        },
        "seq_lengths": [],
        "sparse_times": [],
        "full_times": [],
        "speedups": [],
        "memory_sparse": [],
        "memory_full": [],
        "gate_passes": [],
    }

    for seq_len in args.seq:
        print(f"\nTesting sequence length: {seq_len}")

        # Create input
        x = torch.randn(args.batch, seq_len, args.d_model, device=args.device)

        # Warmup
        print(f"  Warmup ({args.warmup} runs)...")
        for _ in range(args.warmup):
            with torch.no_grad():
                _ = sparse_attn(x)
                _ = full_attn(x, x, x)

        # Benchmark sparse
        print(f"  Benchmarking sparse attention...")
        sparse_times = []
        for i in range(args.n_runs):
            if args.device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            with torch.no_grad():
                sparse_out, sparse_metrics = sparse_attn(x)
            if args.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            sparse_times.append(elapsed)
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{args.n_runs}: {np.mean(sparse_times[-10:]):.4f}s")

        sparse_median = np.median(sparse_times)

        # Benchmark full
        print(f"  Benchmarking full attention...")
        full_times = []
        for i in range(args.n_runs):
            if args.device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            with torch.no_grad():
                full_out, _ = full_attn(x, x, x)
            if args.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            full_times.append(elapsed)
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{args.n_runs}: {np.mean(full_times[-10:]):.4f}s")

        full_median = np.median(full_times)
        speedup = full_median / sparse_median

        # Memory usage (if CUDA)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = sparse_attn(x)
            mem_sparse = torch.cuda.max_memory_allocated() / 1e9

            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = full_attn(x, x, x)
            mem_full = torch.cuda.max_memory_allocated() / 1e9
        else:
            mem_sparse = 0
            mem_full = 0

        # Check gate
        gate_passes = speedup >= 3.0 if seq_len >= 16384 else True

        # Store results
        results["seq_lengths"].append(seq_len)
        results["sparse_times"].append(sparse_median)
        results["full_times"].append(full_median)
        results["speedups"].append(speedup)
        results["memory_sparse"].append(mem_sparse)
        results["memory_full"].append(mem_full)
        results["gate_passes"].append(gate_passes)

        # Print summary
        print(f"\n  Results:")
        print(f"    Sparse:  {sparse_median:.4f}s ({mem_sparse:.2f} GB)")
        print(f"    Full:    {full_median:.4f}s ({mem_full:.2f} GB)")
        print(f"    Speedup: {speedup:.2f}× {'✓' if gate_passes else '✗ GATE FAIL'}")

    # Overall summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for i, seq_len in enumerate(results["seq_lengths"]):
        speedup = results["speedups"][i]
        passes = results["gate_passes"][i]
        status = "✓ PASS" if passes else "✗ FAIL"
        print(f"{seq_len:6d}: {speedup:5.2f}× {status}")

    all_pass = all(results["gate_passes"])
    print("-" * 80)
    print(f"Overall: {'✓ ALL GATES PASS' if all_pass else '✗ SOME GATES FAILED'}")

    # Save report
    with open(args.report, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to: {args.report}")

    # Exit code
    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
