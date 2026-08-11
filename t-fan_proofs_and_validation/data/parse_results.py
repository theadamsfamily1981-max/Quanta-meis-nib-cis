"""Utility to inspect and visualize T-FAN empirical datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt


@dataclass
class Experiment:
    """Structured representation of a single dataset entry."""

    id: str
    phase: int
    configuration: dict
    operating_conditions: dict
    performance: dict
    stability_metrics: dict
    notes: str

    @classmethod
    def from_dict(cls, payload: dict) -> "Experiment":
        return cls(
            id=payload["id"],
            phase=int(payload["phase"]),
            configuration=dict(payload["configuration"]),
            operating_conditions=dict(payload["operating_conditions"]),
            performance=dict(payload["performance"]),
            stability_metrics=dict(payload["stability_metrics"]),
            notes=str(payload.get("notes", "")),
        )


@dataclass
class Dataset:
    metadata: dict
    experiments: List[Experiment]

    @classmethod
    def load(cls, path: Path) -> "Dataset":
        with path.open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
        experiments = [Experiment.from_dict(item) for item in raw.get("experiments", [])]
        return cls(metadata=raw.get("metadata", {}), experiments=experiments)

    def extract_metric(self, metric: str) -> Sequence[float]:
        """Return the requested metric from each experiment.

        The metric name may refer to fields inside the `performance` or
        `stability_metrics` sections. If the metric is not present for an
        experiment, it will be skipped.
        """

        values: List[float] = []
        for experiment in self.experiments:
            for section in (experiment.performance, experiment.stability_metrics):
                if metric in section:
                    values.append(float(section[metric]))
                    break
        return values


def summarize(values: Iterable[float]) -> dict:
    """Compute summary statistics for a list of numerical values."""

    series = list(values)
    if not series:
        raise ValueError("no values were provided for summarization")
    return {
        "count": len(series),
        "mean": mean(series),
        "median": median(series),
        "minimum": min(series),
        "maximum": max(series),
    }


def plot_metric(values: Sequence[float], metric: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(values) + 1), values, marker="o", linestyle="-", color="#1f77b4")
    ax.set_title(f"T-FAN {metric.replace('_', ' ').title()} Across Experiments")
    ax.set_xlabel("Experiment Index")
    ax.set_ylabel(metric.replace("_", " "))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    output_path = output_dir / f"{metric}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to the JSON dataset")
    parser.add_argument(
        "--metric",
        type=str,
        default="efficiency_pct",
        help="Metric to summarize (from performance or stability_metrics)",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures"),
        help="Directory to store generated plots",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = Dataset.load(args.dataset)
    values = dataset.extract_metric(args.metric)
    if not values:
        raise SystemExit(f"Metric '{args.metric}' was not found in the dataset")

    stats = summarize(values)
    print(f"Metric: {args.metric}")
    for key, value in stats.items():
        print(f"  {key:>7}: {value:.3f}" if isinstance(value, float) else f"  {key:>7}: {value}")

    output_path = plot_metric(values, args.metric, args.figures_dir)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
