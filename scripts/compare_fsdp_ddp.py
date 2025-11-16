#!/usr/bin/env python
"""
Compare FSDP vs DDP Performance

Runs both DDP and FSDP benchmarks and compares results.
Checks hard gate: FSDP ≥1.6× speedup vs DDP on 4 GPUs.

Usage:
    # Requires 4 GPUs
    python scripts/compare_fsdp_ddp.py --model-size 7b --batch-size 2

    # CI mode (exit 1 if gate fails)
    python scripts/compare_fsdp_ddp.py --model-size 7b --ci
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_benchmark(mode: str, model_size: str, batch_size: int, num_gpus: int) -> dict:
    """Run benchmark using torchrun."""
    output_file = f"artifacts/bench_{mode}_{model_size}.json"

    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "benchmarks/bench_fsdp_vs_ddp.py",
        "--mode", mode,
        "--model-size", model_size,
        "--batch-size", str(batch_size),
        "--iterations", "20",
        "--output", output_file
    ]

    print(f"\nRunning {mode.upper()} benchmark...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"✗ {mode.upper()} benchmark failed:")
        print(result.stderr)
        return None

    # Load results
    try:
        with open(output_file, 'r') as f:
            stats = json.load(f)
        return stats
    except Exception as e:
        print(f"✗ Failed to load {mode} results: {e}")
        return None


def compare_results(ddp_stats: dict, fsdp_stats: dict) -> dict:
    """Compare DDP and FSDP results."""
    if ddp_stats is None or fsdp_stats is None:
        return {'error': 'Missing benchmark results'}

    # Compute speedup
    ddp_time = ddp_stats['time_per_step_ms']
    fsdp_time = fsdp_stats['time_per_step_ms']
    speedup = ddp_time / fsdp_time

    # Memory comparison
    ddp_mem = ddp_stats.get('peak_memory_gb', 0)
    fsdp_mem = fsdp_stats.get('peak_memory_gb', 0)
    mem_reduction = (ddp_mem - fsdp_mem) / ddp_mem if ddp_mem > 0 else 0

    # Throughput comparison
    ddp_throughput = ddp_stats.get('throughput_samples_per_sec', 0)
    fsdp_throughput = fsdp_stats.get('throughput_samples_per_sec', 0)
    throughput_gain = fsdp_throughput / ddp_throughput if ddp_throughput > 0 else 0

    return {
        'ddp': {
            'time_per_step_ms': ddp_time,
            'throughput': ddp_throughput,
            'memory_gb': ddp_mem
        },
        'fsdp': {
            'time_per_step_ms': fsdp_time,
            'throughput': fsdp_throughput,
            'memory_gb': fsdp_mem
        },
        'speedup': speedup,
        'throughput_gain': throughput_gain,
        'memory_reduction': mem_reduction,
        'passes_speedup_gate': speedup >= 1.6,
        'gate_threshold': 1.6
    }


def print_comparison(comparison: dict):
    """Print comparison results."""
    print("\n" + "=" * 60)
    print("FSDP vs DDP Comparison")
    print("=" * 60)

    if 'error' in comparison:
        print(f"✗ Error: {comparison['error']}")
        return

    print(f"\nDDP:")
    print(f"  Time/step:   {comparison['ddp']['time_per_step_ms']:.2f} ms")
    print(f"  Throughput:  {comparison['ddp']['throughput']:.2f} samples/sec")
    print(f"  Memory:      {comparison['ddp']['memory_gb']:.2f} GB")

    print(f"\nFSDP:")
    print(f"  Time/step:   {comparison['fsdp']['time_per_step_ms']:.2f} ms")
    print(f"  Throughput:  {comparison['fsdp']['throughput']:.2f} samples/sec")
    print(f"  Memory:      {comparison['fsdp']['memory_gb']:.2f} GB")

    print(f"\nGains:")
    print(f"  Speedup:          {comparison['speedup']:.2f}×")
    print(f"  Throughput gain:  {comparison['throughput_gain']:.2f}×")
    print(f"  Memory reduction: {comparison['memory_reduction']:.1%}")

    print(f"\nGate Check:")
    status = '✓' if comparison['passes_speedup_gate'] else '✗'
    print(f"  {status} Speedup ≥{comparison['gate_threshold']}×: {comparison['speedup']:.2f}×")

    if comparison['passes_speedup_gate']:
        print(f"\n✓ PASSED: FSDP achieves {comparison['speedup']:.2f}× speedup vs DDP")
    else:
        print(f"\n✗ FAILED: FSDP only achieves {comparison['speedup']:.2f}× speedup (target: ≥{comparison['gate_threshold']}×)")


def main():
    parser = argparse.ArgumentParser(description="Compare FSDP vs DDP")
    parser.add_argument("--model-size", type=str, default="7b",
                        choices=["1b", "3b", "7b", "13b"],
                        help="Model size to benchmark")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size per GPU")
    parser.add_argument("--num-gpus", type=int, default=4,
                        help="Number of GPUs to use")
    parser.add_argument("--output", type=str,
                        help="Output JSON file for comparison")
    parser.add_argument("--ci", action="store_true",
                        help="CI mode (exit 1 if gate fails)")
    parser.add_argument("--skip-ddp", action="store_true",
                        help="Skip DDP benchmark (use cached results)")
    parser.add_argument("--skip-fsdp", action="store_true",
                        help="Skip FSDP benchmark (use cached results)")

    args = parser.parse_args()

    # Create artifacts directory
    Path("artifacts").mkdir(exist_ok=True)

    # Run DDP benchmark
    if not args.skip_ddp:
        ddp_stats = run_benchmark("ddp", args.model_size, args.batch_size, args.num_gpus)
    else:
        # Load cached results
        try:
            with open(f"artifacts/bench_ddp_{args.model_size}.json", 'r') as f:
                ddp_stats = json.load(f)
            print("✓ Loaded cached DDP results")
        except Exception as e:
            print(f"✗ Failed to load cached DDP results: {e}")
            ddp_stats = None

    # Run FSDP benchmark
    if not args.skip_fsdp:
        fsdp_stats = run_benchmark("fsdp", args.model_size, args.batch_size, args.num_gpus)
    else:
        # Load cached results
        try:
            with open(f"artifacts/bench_fsdp_{args.model_size}.json", 'r') as f:
                fsdp_stats = json.load(f)
            print("✓ Loaded cached FSDP results")
        except Exception as e:
            print(f"✗ Failed to load cached FSDP results: {e}")
            fsdp_stats = None

    # Compare results
    comparison = compare_results(ddp_stats, fsdp_stats)

    # Print comparison
    print_comparison(comparison)

    # Save comparison
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\n✓ Saved comparison to {args.output}")

    # CI mode: exit with error if gate fails
    if args.ci:
        if comparison.get('passes_speedup_gate', False):
            print("\n✓ CI PASSED")
            sys.exit(0)
        else:
            print("\n✗ CI FAILED")
            sys.exit(1)


if __name__ == "__main__":
    main()
