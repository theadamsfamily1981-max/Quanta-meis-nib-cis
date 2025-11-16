#!/usr/bin/env python
"""
Benchmark Fused Sparse-Radial Attention

Comprehensive benchmark comparing fused CUDA kernel vs unfused PyTorch.

Hard gates:
- ≥2× speedup
- <1e-3 numerical error
- -20% VRAM usage

Usage:
    # Run benchmark with default settings
    python benchmarks/bench_fused_radial.py

    # Sweep across sequence lengths
    python benchmarks/bench_fused_radial.py --sweep-seq-len

    # Export results to JSON
    python benchmarks/bench_fused_radial.py --output results.json

    # CI mode (exit 1 if gates fail)
    python benchmarks/bench_fused_radial.py --ci
"""

import argparse
import json
import sys
from pathlib import Path
import time

import torch
import numpy as np

# Add tfan to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tfan.kernels.fused_radial_attn import (
    FusedRadialAttention,
    benchmark_fused_vs_unfused,
    CUDA_AVAILABLE
)


def sweep_seq_lengths(
    seq_lengths: list = [128, 256, 512, 1024, 2048],
    batch_size: int = 4,
    num_heads: int = 8,
    head_dim: int = 64,
    num_landmarks: int = 16,
    avg_radius: int = 32
) -> list:
    """Benchmark across different sequence lengths."""
    results = []

    for seq_len in seq_lengths:
        print(f"\nBenchmarking seq_len={seq_len}...")

        stats = benchmark_fused_vs_unfused(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            num_landmarks=num_landmarks,
            avg_radius=avg_radius,
            num_iterations=50
        )

        stats['seq_len'] = seq_len
        results.append(stats)

        print(f"  Speedup: {stats['speedup']:.2f}×")
        print(f"  Error:   {stats['max_error']:.6f}")
        print(f"  Memory:  {stats['mem_reduction']:.1%} reduction")

    return results


def sweep_sparsity(
    radii: list = [8, 16, 32, 64, 128],
    batch_size: int = 4,
    num_heads: int = 8,
    seq_len: int = 512,
    head_dim: int = 64,
    num_landmarks: int = 16
) -> list:
    """Benchmark across different sparsity levels (radii)."""
    results = []

    for avg_radius in radii:
        print(f"\nBenchmarking avg_radius={avg_radius}...")

        stats = benchmark_fused_vs_unfused(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            num_landmarks=num_landmarks,
            avg_radius=avg_radius,
            num_iterations=50
        )

        stats['avg_radius'] = avg_radius
        results.append(stats)

        print(f"  Speedup: {stats['speedup']:.2f}×")
        print(f"  Error:   {stats['max_error']:.6f}")
        print(f"  Memory:  {stats['mem_reduction']:.1%} reduction")

    return results


def check_gates(stats: dict) -> dict:
    """
    Check hard gates.

    Gates:
    - Speedup ≥2×
    - Error <1e-3
    - Memory reduction ≥20%

    Returns:
        gate_results: Dict with pass/fail for each gate
    """
    gates = {
        'speedup': {
            'value': stats.get('speedup', 0),
            'threshold': 2.0,
            'pass': stats.get('passes_speedup_gate', False)
        },
        'error': {
            'value': stats.get('max_error', 1.0),
            'threshold': 1e-3,
            'pass': stats.get('passes_error_gate', False)
        },
        'memory': {
            'value': stats.get('mem_reduction', 0),
            'threshold': 0.20,
            'pass': stats.get('passes_mem_gate', False)
        }
    }

    gates['overall'] = {
        'pass': all(g['pass'] for g in gates.values() if 'pass' in g)
    }

    return gates


def print_summary(results: list):
    """Print summary statistics across all runs."""
    if not results:
        return

    speedups = [r['speedup'] for r in results if r.get('speedup', 0) > 0]
    errors = [r['max_error'] for r in results]
    mem_reductions = [r['mem_reduction'] for r in results if r.get('mem_reduction', 0) > 0]

    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    if speedups:
        print(f"\nSpeedup:")
        print(f"  Mean: {np.mean(speedups):.2f}×")
        print(f"  Min:  {np.min(speedups):.2f}×")
        print(f"  Max:  {np.max(speedups):.2f}×")

    print(f"\nNumerical Error:")
    print(f"  Mean: {np.mean(errors):.6f}")
    print(f"  Max:  {np.max(errors):.6f}")

    if mem_reductions:
        print(f"\nMemory Reduction:")
        print(f"  Mean: {np.mean(mem_reductions):.1%}")
        print(f"  Min:  {np.min(mem_reductions):.1%}")
        print(f"  Max:  {np.max(mem_reductions):.1%}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark fused radial attention")
    parser.add_argument('--batch-size', type=int, default=4, help="Batch size")
    parser.add_argument('--num-heads', type=int, default=8, help="Number of attention heads")
    parser.add_argument('--seq-len', type=int, default=512, help="Sequence length")
    parser.add_argument('--head-dim', type=int, default=64, help="Head dimension")
    parser.add_argument('--num-landmarks', type=int, default=16, help="Number of landmarks per query")
    parser.add_argument('--avg-radius', type=int, default=32, help="Average radial distance")
    parser.add_argument('--iterations', type=int, default=100, help="Number of iterations")
    parser.add_argument('--sweep-seq-len', action='store_true', help="Sweep sequence lengths")
    parser.add_argument('--sweep-sparsity', action='store_true', help="Sweep sparsity (radii)")
    parser.add_argument('--output', type=str, help="Output JSON file")
    parser.add_argument('--ci', action='store_true', help="CI mode (exit 1 on gate failure)")

    args = parser.parse_args()

    print("=" * 60)
    print("Fused Sparse-Radial Attention Benchmark")
    print("=" * 60)

    if not CUDA_AVAILABLE:
        print("\n⚠ CUDA kernel not available, running unfused baseline only")
        if args.ci:
            print("✗ CI FAILED: CUDA kernel required")
            sys.exit(1)

    results = []

    if args.sweep_seq_len:
        results = sweep_seq_lengths(
            batch_size=args.batch_size,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            num_landmarks=args.num_landmarks,
            avg_radius=args.avg_radius
        )

    elif args.sweep_sparsity:
        results = sweep_sparsity(
            batch_size=args.batch_size,
            num_heads=args.num_heads,
            seq_len=args.seq_len,
            head_dim=args.head_dim,
            num_landmarks=args.num_landmarks
        )

    else:
        # Single run
        print(f"\nConfiguration:")
        print(f"  Batch size:    {args.batch_size}")
        print(f"  Num heads:     {args.num_heads}")
        print(f"  Sequence len:  {args.seq_len}")
        print(f"  Head dim:      {args.head_dim}")
        print(f"  Landmarks:     {args.num_landmarks}")
        print(f"  Avg radius:    {args.avg_radius}")
        print(f"  Iterations:    {args.iterations}")

        stats = benchmark_fused_vs_unfused(
            batch_size=args.batch_size,
            num_heads=args.num_heads,
            seq_len=args.seq_len,
            head_dim=args.head_dim,
            num_landmarks=args.num_landmarks,
            avg_radius=args.avg_radius,
            num_iterations=args.iterations
        )

        results = [stats]

        print(f"\nTiming:")
        print(f"  Fused:   {stats['time_fused_ms']:.3f} ms")
        print(f"  Unfused: {stats['time_unfused_ms']:.3f} ms")
        print(f"  Speedup: {stats['speedup']:.2f}×")

        print(f"\nAccuracy:")
        print(f"  Max error: {stats['max_error']:.6f}")

        print(f"\nMemory:")
        print(f"  Fused:     {stats['mem_fused_mb']:.2f} MB")
        print(f"  Unfused:   {stats['mem_unfused_mb']:.2f} MB")
        print(f"  Reduction: {stats['mem_reduction']:.1%}")

    # Print summary if multiple runs
    if len(results) > 1:
        print_summary(results)

    # Check gates
    print("\n" + "=" * 60)
    print("Gate Checks")
    print("=" * 60)

    # Use first result for gate checking (or aggregate if sweep)
    gate_stats = results[0] if results else {}
    gates = check_gates(gate_stats)

    print(f"\n{'✓' if gates['speedup']['pass'] else '✗'} Speedup ≥2×: {gates['speedup']['value']:.2f}×")
    print(f"{'✓' if gates['error']['pass'] else '✗'} Error <1e-3: {gates['error']['value']:.6f}")
    print(f"{'✓' if gates['memory']['pass'] else '✗'} Memory -20%: {gates['memory']['value']:.1%}")

    print(f"\n{'✓' if gates['overall']['pass'] else '✗'} Overall: {'PASS' if gates['overall']['pass'] else 'FAIL'}")

    # Export to JSON
    if args.output:
        output_data = {
            'results': results,
            'gates': gates,
            'config': vars(args)
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Exported results to {args.output}")

    # CI mode: exit with error code if gates fail
    if args.ci:
        if not gates['overall']['pass']:
            print("\n✗ CI FAILED: Hard gates not met")
            sys.exit(1)
        else:
            print("\n✓ CI PASSED: All gates met")
            sys.exit(0)


if __name__ == '__main__':
    main()
