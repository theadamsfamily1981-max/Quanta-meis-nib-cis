"""Utilities for validating metric outputs against configured thresholds."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Mapping, MutableMapping


class ThresholdViolation(Exception):
    """Raised when one or more metrics fall outside configured thresholds."""


def load_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text())


def validate_metrics(
    metrics: Mapping[str, float],
    thresholds: Mapping[str, Mapping[str, float]],
    suite_name: str,
    failures: MutableMapping[str, Dict[str, float]],
) -> None:
    for key, rules in thresholds.items():
        if key not in metrics:
            failures.setdefault(suite_name, {})[key] = float("nan")
            continue
        value = float(metrics[key])
        minimum = rules.get("min")
        maximum = rules.get("max")
        if minimum is not None and value < minimum:
            failures.setdefault(suite_name, {})[key] = value
        if maximum is not None and value > maximum:
            failures.setdefault(suite_name, {})[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate metrics against JSON thresholds.")
    parser.add_argument("--comp", type=Path, required=True, help="Path to the comprehensive metrics JSON file.")
    parser.add_argument("--imp", type=Path, required=True, help="Path to the impossibility metrics JSON file.")
    parser.add_argument("--thresholds", type=Path, required=True, help="Threshold configuration JSON file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comprehensive = load_json(args.comp)
    impossibility = load_json(args.imp)
    thresholds = load_json(args.thresholds)

    failures: Dict[str, Dict[str, float]] = {}
    validate_metrics(
        metrics=comprehensive.get("metrics", {}),
        thresholds=thresholds.get("comprehensive", {}),
        suite_name="comprehensive",
        failures=failures,
    )
    validate_metrics(
        metrics=impossibility.get("metrics", {}),
        thresholds=thresholds.get("impossibility", {}),
        suite_name="impossibility",
        failures=failures,
    )

    if failures:
        print("Validation thresholds failed:")
        for suite, metrics in failures.items():
            for metric, value in metrics.items():
                print(f"- {suite}.{metric}: observed={value}")
        raise ThresholdViolation("Validation thresholds not met")

    print("All validation thresholds satisfied.")


if __name__ == "__main__":
    try:
        main()
    except ThresholdViolation as exc:
        sys.exit(str(exc))
