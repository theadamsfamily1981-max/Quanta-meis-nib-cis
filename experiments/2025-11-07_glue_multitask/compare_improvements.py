#!/usr/bin/env python
"""Compare per-task improvements derived from GLUE multitask results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GLUE multitask improvements")
    parser.add_argument("results", type=Path, help="Path to results_glue_multitask.json")
    return parser.parse_args()


def compute_improvements(results: Dict[str, object]) -> Dict[str, object]:
    config = results.get("config")
    tasks_config: List[str]
    if isinstance(config, dict) and isinstance(config.get("tasks"), list):
        tasks_config = [str(task) for task in config["tasks"]]
    else:
        tasks_config = []

    tasks = results.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("Results JSON is missing task metrics")

    ordered_tasks = tasks_config or sorted(tasks.keys())
    baseline_name = ordered_tasks[0]
    baseline_metrics = tasks.get(baseline_name)
    if not isinstance(baseline_metrics, dict):
        raise ValueError(f"Baseline task '{baseline_name}' metrics are unavailable")

    baseline_eval = float(baseline_metrics.get("eval_accuracy", 0.0))

    improvements: Dict[str, Dict[str, float]] = {}
    for name in ordered_tasks:
        metrics = tasks.get(name)
        if not isinstance(metrics, dict):
            raise ValueError(f"Task metrics for {name} must be a dictionary")
        eval_acc = float(metrics.get("eval_accuracy", 0.0))
        train_acc = float(metrics.get("train_accuracy", 0.0))
        improvements[name] = {
            "delta_vs_baseline_eval": eval_acc - baseline_eval,
            "overfit_gap": eval_acc - train_acc,
        }

    summary = {
        "baseline_task": baseline_name,
        "baseline_eval_accuracy": baseline_eval,
        "improvements": improvements,
    }
    return summary


def main() -> int:
    args = parse_args()
    results_path = args.results
    if not results_path.exists():
        raise SystemExit(f"Results file not found: {results_path}")

    results = json.loads(results_path.read_text())
    summary = compute_improvements(results)

    output_path = results_path.with_name("results_glue_multitask_improvements.json")
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote improvement summary to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
