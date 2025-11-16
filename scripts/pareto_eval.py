#!/usr/bin/env python
"""
Pareto Auto-Runner Evaluation

Batch runs configurations and computes Pareto frontier.

Hard gates:
- ≥6 non-dominated points with crowding ε=0.05
- End-to-end wall-time ≤6h on 1×RTX 3090

Usage:
    # Grid search
    python scripts/pareto_eval.py \
        --mode grid \
        --params k_landmarks:32,64,128 local_window:128,256,512 \
        --objectives latency_ms,accuracy,memory_mb

    # Bayesian optimization with EHVI
    python scripts/pareto_eval.py \
        --mode bo \
        --n-initial 10 \
        --n-iterations 50 \
        --objectives latency_ms,accuracy

    # With real runner
    python scripts/pareto_eval.py \
        --mode grid \
        --runner-script scripts/train_model.py \
        --objectives latency_ms,accuracy
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add tfan to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tfan.pareto import (
    ParetoRunner,
    RunConfig,
    pareto_frontier,
    hypervolume,
    check_crowding_gate,
    summarize_results,
    export_dashboard,
    generate_html_report
)
from tfan.pareto.runner import mock_runner_fn


def parse_param_spec(spec: str) -> dict:
    """
    Parse parameter specification.

    Format: "param1:val1,val2,val3 param2:val1,val2"

    Returns:
        Dict mapping param names to value lists
    """
    param_grid = {}

    for param_spec in spec.split():
        if ':' not in param_spec:
            continue

        param_name, values_str = param_spec.split(':', 1)
        values = values_str.split(',')

        # Try to convert to numbers
        parsed_values = []
        for v in values:
            try:
                # Try int first
                parsed_values.append(int(v))
            except ValueError:
                try:
                    # Try float
                    parsed_values.append(float(v))
                except ValueError:
                    # Keep as string
                    parsed_values.append(v)

        param_grid[param_name] = parsed_values

    return param_grid


def check_gates(frontier, results, wall_time_h):
    """
    Check hard gates.

    Gates:
    - ≥6 non-dominated points
    - Crowding ε=0.05
    - Wall-time ≤6h

    Returns:
        gates: Dict with gate results
    """
    gates = {}

    # Gate 1: Number of frontier points
    num_frontier = len(frontier)
    gates['num_frontier'] = {
        'value': num_frontier,
        'threshold': 6,
        'pass': num_frontier >= 6
    }

    status = '✓' if num_frontier >= 6 else '✗'
    print(f"\n{status} Frontier points: {num_frontier} (target: ≥6)")

    # Gate 2: Crowding distance
    crowding_pass = check_crowding_gate(frontier, epsilon=0.05)
    gates['crowding'] = {
        'value': crowding_pass,
        'threshold': True,
        'pass': crowding_pass
    }

    status = '✓' if crowding_pass else '✗'
    print(f"{status} Crowding distance: ε=0.05 {'passed' if crowding_pass else 'failed'}")

    # Gate 3: Wall-time
    gates['wall_time'] = {
        'value': wall_time_h,
        'threshold': 6.0,
        'pass': wall_time_h <= 6.0
    }

    status = '✓' if wall_time_h <= 6.0 else '✗'
    print(f"{status} Wall-time: {wall_time_h:.2f}h (target: ≤6h)")

    # Overall
    all_pass = all(g['pass'] for g in gates.values())
    gates['overall'] = {'pass': all_pass}

    return gates


def main():
    parser = argparse.ArgumentParser(description="Pareto Auto-Runner Evaluation")
    parser.add_argument('--mode', type=str, default='grid', choices=['grid', 'bo'],
                        help="Optimization mode")
    parser.add_argument('--params', type=str,
                        default='k_landmarks:32,64,128 local_window:128,256,512',
                        help="Parameter grid specification")
    parser.add_argument('--objectives', type=str, default='latency_ms,accuracy,memory_mb',
                        help="Comma-separated objective names")
    parser.add_argument('--minimize', type=str, default='latency_ms,memory_mb',
                        help="Objectives to minimize")
    parser.add_argument('--maximize', type=str, default='accuracy',
                        help="Objectives to maximize")
    parser.add_argument('--runner-script', type=str, help="Custom runner script")
    parser.add_argument('--max-workers', type=int, default=1,
                        help="Number of parallel workers")
    parser.add_argument('--output-dir', type=str, default='artifacts/pareto',
                        help="Output directory")
    parser.add_argument('--max-time-h', type=float, default=6.0,
                        help="Maximum wall-time in hours")

    args = parser.parse_args()

    # Parse objectives
    objectives = [obj.strip() for obj in args.objectives.split(',')]
    minimize = [obj.strip() for obj in args.minimize.split(',') if obj.strip()]
    maximize = [obj.strip() for obj in args.maximize.split(',') if obj.strip()]

    print(f"\n{'='*60}")
    print("Pareto Auto-Runner")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Objectives: {objectives}")
    print(f"Minimize: {minimize}")
    print(f"Maximize: {maximize}")
    print(f"Max wall-time: {args.max_time_h}h")
    print(f"{'='*60}\n")

    # Initialize runner
    runner = ParetoRunner(
        objectives=objectives,
        artifacts_dir=args.output_dir,
        max_time_per_run=3600.0  # 1 hour per run
    )

    # Parse parameter grid
    param_grid = parse_param_spec(args.params)

    print(f"Parameter grid: {param_grid}")

    # Generate configs
    if args.mode == 'grid':
        configs = runner.grid_search(param_grid)
    else:
        # Bayesian optimization (simplified - would use real BO library)
        print("⚠ Bayesian optimization not yet implemented, falling back to grid search")
        configs = runner.grid_search(param_grid)

    # Use mock runner or custom script
    if args.runner_script:
        print(f"⚠ Custom runner script not yet implemented: {args.runner_script}")
        print("Using mock runner for demonstration")
        runner_fn = mock_runner_fn
    else:
        runner_fn = mock_runner_fn

    # Run batch
    start_time = time.perf_counter()

    results = runner.run_batch(
        configs=configs,
        runner_fn=runner_fn,
        max_workers=args.max_workers
    )

    wall_time_s = time.perf_counter() - start_time
    wall_time_h = wall_time_s / 3600

    # Compute Pareto frontier
    print(f"\n{'='*60}")
    print("Computing Pareto Frontier")
    print(f"{'='*60}")

    frontier = pareto_frontier(
        results=results,
        minimize=minimize,
        maximize=maximize
    )

    # Compute hypervolume
    if frontier:
        # Reference point (worst acceptable values)
        reference = []
        for obj in minimize:
            reference.append(max(r.metrics.get(obj, 0) for r in results if r.status == 'success'))
        for obj in maximize:
            reference.append(-min(r.metrics.get(obj, 0) for r in results if r.status == 'success'))

        reference_point = [r * 1.1 for r in reference]  # 10% margin

        hv = hypervolume(frontier, reference_point)
        print(f"✓ Hypervolume: {hv:.4f}")

    # Check gates
    gates = check_gates(frontier, results, wall_time_h)

    # Summarize
    summary = summarize_results(results, frontier, minimize, maximize)
    summary['gates'] = gates
    summary['wall_time_h'] = wall_time_h

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for key, value in summary.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")

    # Export dashboard
    dashboard_path = Path(args.output_dir) / 'dashboard.json'
    export_dashboard(frontier, results, str(dashboard_path), minimize, maximize)

    # Generate HTML report
    html_path = Path(args.output_dir) / 'report.html'
    generate_html_report(frontier, results, str(html_path), minimize, maximize)

    # Save summary
    summary_path = Path(args.output_dir) / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ Results saved to {args.output_dir}")
    print(f"  Dashboard: {dashboard_path}")
    print(f"  Report: {html_path}")
    print(f"  Summary: {summary_path}")
    print(f"{'='*60}")

    # Exit with appropriate code
    if gates['overall']['pass']:
        print("\n✓ All gates PASSED")
        sys.exit(0)
    else:
        print("\n✗ Some gates FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
