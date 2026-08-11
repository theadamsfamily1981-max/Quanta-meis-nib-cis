#!/usr/bin/env python
"""Post-process GLUE multitask results to compute aggregate metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze GLUE multitask results")
    parser.add_argument("results", type=Path, help="Path to results_glue_multitask.json")
    return parser.parse_args()


def compute_summary(results: Dict[str, object]) -> Dict[str, object]:
    tasks = results.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("Results JSON is missing the 'tasks' dictionary")

    total_eval = 0.0
    total_train = 0.0
    best_name = None
    best_eval = float("-inf")
    per_task = {}

    for name, metrics in tasks.items():
        if not isinstance(metrics, dict):
            raise ValueError(f"Task metrics for {name} must be a dictionary")
        eval_acc = float(metrics.get("eval_accuracy", 0.0))
        train_acc = float(metrics.get("train_accuracy", 0.0))
        total_eval += eval_acc
        total_train += train_acc
        if eval_acc > best_eval:
            best_name = name
            best_eval = eval_acc
        per_task[name] = {
            "eval_accuracy": eval_acc,
            "train_accuracy": train_acc,
            "eval_loss": float(metrics.get("eval_loss", 0.0)),
            "num_eval_examples": float(metrics.get("num_eval_examples", 0.0)),
        }

    num_tasks = len(per_task)
    average_eval = total_eval / max(num_tasks, 1)
    average_train = total_train / max(num_tasks, 1)

    return {
        "num_tasks": num_tasks,
        "average_eval_accuracy": average_eval,
        "average_train_accuracy": average_train,
        "best_task": {
            "name": best_name,
            "eval_accuracy": best_eval,
        },
        "tasks": per_task,
    }


def main() -> int:
    args = parse_args()
    results_path = args.results
    if not results_path.exists():
        raise SystemExit(f"Results file not found: {results_path}")

    results = json.loads(results_path.read_text())
    summary = compute_summary(results)

    summary_path = results_path.with_name("results_glue_multitask_analysis.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote analysis summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
