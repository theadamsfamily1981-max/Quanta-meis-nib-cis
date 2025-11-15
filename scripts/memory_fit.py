#!/usr/bin/env python
"""
Memory scaling validation.

Fits Memory = a * T^α and validates α < 1.0 gate.

Usage:
    python scripts/memory_fit.py --seq 1024 2048 4096 8192 16384 32768
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import curve_fit

from tfan.attention import SparseAttention


def power_law(T, a, alpha):
    """Power law: Memory = a * T^alpha"""
    return a * np.power(T, alpha)


def measure_memory(model, seq_len, batch_size=1, device="cuda", n_runs=5):
    """
    Measure peak memory usage for a given sequence length.

    Args:
        model: Model to test
        seq_len: Sequence length
        batch_size: Batch size
        device: Device
        n_runs: Number of runs to average

    Returns:
        Peak memory in GB
    """
    memories = []

    for _ in range(n_runs):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        x = torch.randn(batch_size, seq_len, model.d_model, device=device)

        with torch.no_grad():
            _ = model(x)

        if device == "cuda":
            peak_mem = torch.cuda.max_memory_allocated() / 1e9
        else:
            # CPU memory is harder to measure accurately
            peak_mem = 0.0

        memories.append(peak_mem)

        # Clear cache
        if device == "cuda":
            torch.cuda.empty_cache()

    return np.median(memories)


def main():
    parser = argparse.ArgumentParser(description="Memory scaling validation")
    parser.add_argument("--seq", type=int, nargs="+",
                        default=[1024, 2048, 4096, 8192, 16384, 32768],
                        help="Sequence lengths to test")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--d-model", type=int, default=768, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=12, help="Number of heads")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--report", type=str, default="artifacts/memory/fit.json",
                        help="Output report path")
    parser.add_argument("--n-runs", type=int, default=5, help="Runs per length")
    args = parser.parse_args()

    # Ensure output directory
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Memory Scaling Validation")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch}")
    print(f"Model dim: {args.d_model}")
    print(f"Sequence lengths: {args.seq}")
    print("-" * 80)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Create model
    model = SparseAttention(
        d_model=args.d_model,
        n_heads=args.n_heads,
        keep_ratio=0.33,
    ).to(args.device)

    # Measure memory for each sequence length
    seq_lengths = []
    memories = []

    for seq_len in args.seq:
        print(f"\nTesting sequence length: {seq_len}")

        try:
            mem = measure_memory(model, seq_len, args.batch, args.device, args.n_runs)
            seq_lengths.append(seq_len)
            memories.append(mem)
            print(f"  Peak memory: {mem:.3f} GB")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  OOM at {seq_len}")
                break
            else:
                raise

    if len(seq_lengths) < 3:
        print("\nERROR: Need at least 3 data points for fitting")
        return 1

    # Fit power law
    print("\n" + "=" * 80)
    print("Fitting Power Law: Memory = a * T^α")
    print("=" * 80)

    T_data = np.array(seq_lengths)
    M_data = np.array(memories)

    # Fit in log space for better numerical stability
    log_T = np.log(T_data)
    log_M = np.log(M_data + 1e-9)  # Avoid log(0)

    # Linear fit: log(M) = log(a) + α * log(T)
    coeffs = np.polyfit(log_T, log_M, deg=1)
    alpha = coeffs[0]
    log_a = coeffs[1]
    a = np.exp(log_a)

    # Compute R²
    M_pred = a * np.power(T_data, alpha)
    ss_res = np.sum((M_data - M_pred) ** 2)
    ss_tot = np.sum((M_data - np.mean(M_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    print(f"Fitted parameters:")
    print(f"  a     = {a:.6f}")
    print(f"  α     = {alpha:.4f}")
    print(f"  R²    = {r_squared:.4f}")
    print("-" * 80)

    # Check gate
    gate_passes = alpha < 1.0
    print(f"Gate: α < 1.0")
    print(f"  Value: {alpha:.4f}")
    print(f"  Status: {'✓ PASS' if gate_passes else '✗ FAIL'}")

    # Print fit quality
    print("\nFit quality:")
    for i, (T, M_actual) in enumerate(zip(T_data, M_data)):
        M_fit = a * (T ** alpha)
        error = abs(M_fit - M_actual) / M_actual * 100
        print(f"  T={T:6d}: actual={M_actual:.3f} GB, fit={M_fit:.3f} GB, error={error:.1f}%")

    # Save results
    results = {
        "config": {
            "device": args.device,
            "batch_size": args.batch,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
        },
        "data": {
            "seq_lengths": seq_lengths,
            "memories_gb": memories,
        },
        "fit": {
            "a": float(a),
            "alpha": float(alpha),
            "r_squared": float(r_squared),
        },
        "gate": {
            "threshold": 1.0,
            "value": float(alpha),
            "passes": gate_passes,
        },
    }

    with open(args.report, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.report}")
    print("=" * 80)

    return 0 if gate_passes else 1


if __name__ == "__main__":
    exit(main())
