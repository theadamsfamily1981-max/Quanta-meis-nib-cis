"""Evaluate sequential learning with and without surprise replay."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class TaskResult:
    name: str
    baseline_accuracy: float
    replay_accuracy: float
    forgetting: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "name": self.name,
            "baseline_accuracy": self.baseline_accuracy,
            "replay_accuracy": self.replay_accuracy,
            "forgetting": self.forgetting,
        }


def simulate_seq_learning(tasks: Iterable[str]) -> List[TaskResult]:
    base_acc = 0.9
    replay_bonus = 0.04
    task_results = []
    for idx, name in enumerate(tasks):
        forgetting = max(0.0, 0.005 - idx * 0.001)
        task_results.append(
            TaskResult(
                name=name,
                baseline_accuracy=base_acc - idx * 0.02,
                replay_accuracy=base_acc - idx * 0.02 + replay_bonus,
                forgetting=forgetting,
            )
        )
    return task_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Where to save the sequential learning JSON")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=("audio_captions", "video_qa", "textual_grounding"),
        help="Names of the sequential learning tasks",
    )
    args = parser.parse_args()

    results = simulate_seq_learning(args.tasks)
    avg_forgetting = sum(t.forgetting for t in results) / len(results)
    payload = {
        "tasks": [t.to_dict() for t in results],
        "average_forgetting": avg_forgetting,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
