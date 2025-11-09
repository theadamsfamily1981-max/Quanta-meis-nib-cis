"""Generate Phase 1 figures from the T-FAN dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


def load_phase1(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def plot_metric(entries: Sequence[dict], metric: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    x = [item["id"] for item in entries]
    y = [item[metric] for item in entries]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(x, y, color="#2ca02c")
    ax.set_title(f"Phase 1 {metric.replace('_', ' ').title()}")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_xlabel("Experiment ID")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    output_path = output_dir / f"phase1_{metric}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    data = load_phase1(Path("phase1_experiments.json"))
    entries = data["entries"]
    figures_dir = Path("figures")

    for metric in ("efficiency_pct", "torque_nm", "stall_margin_pct"):
        path = plot_metric(entries, metric, figures_dir)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
