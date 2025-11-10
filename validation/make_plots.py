"""Generate plots and summary assets for validation results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping

import matplotlib.pyplot as plt


def load_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text())


def plot_metrics(metrics: Mapping[str, float], title: str, path: Path) -> None:
    keys = list(metrics.keys())
    if not keys:
        figure = plt.figure(figsize=(6, 3))
        axis = figure.add_subplot(1, 1, 1)
        axis.text(0.5, 0.5, "No metrics available", ha="center", va="center")
        axis.set_axis_off()
        figure.tight_layout()
        figure.savefig(path)
        plt.close(figure)
        return

    values = [float(metrics[key]) for key in keys]
    indices = list(range(len(keys)))
    figure = plt.figure(figsize=(10, 4))
    axis = figure.add_subplot(1, 1, 1)
    axis.bar(indices, values, color="#3c82f6")
    axis.set_title(title)
    axis.set_ylabel("Metric value")
    axis.set_xticks(indices)
    axis.set_xticklabels(keys, rotation=45, ha="right")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def build_report(comp_path: Path, imp_path: Path, outdir: Path) -> None:
    comprehensive = load_json(comp_path)
    impossibility = load_json(imp_path)
    metrics = {
        "comprehensive": comprehensive.get("metrics", {}),
        "impossibility": impossibility.get("metrics", {}),
    }

    outdir.mkdir(parents=True, exist_ok=True)

    plot_metrics(metrics["comprehensive"], "Comprehensive Validation Metrics", outdir / "comprehensive.png")
    plot_metrics(metrics["impossibility"], "Impossibility Validation Metrics", outdir / "impossibility.png")

    summary_path = outdir / "summary.json"
    summary_payload: Dict[str, object] = {
        "comprehensive": comprehensive,
        "impossibility": impossibility,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    markdown_path = outdir / "report.md"
    markdown_content = """# Validation Report

## Comprehensive Suite
![Comprehensive metrics](comprehensive.png)

## Impossibility Suite
![Impossibility metrics](impossibility.png)
"""
    markdown_path.write_text(markdown_content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots and summary assets for validation results.")
    parser.add_argument("--comp", type=Path, required=True, help="Path to the comprehensive metrics JSON file.")
    parser.add_argument("--imp", type=Path, required=True, help="Path to the impossibility metrics JSON file.")
    parser.add_argument("--outdir", type=Path, required=True, help="Directory to write the generated plots and report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_report(args.comp, args.imp, args.outdir)


if __name__ == "__main__":
    main()
