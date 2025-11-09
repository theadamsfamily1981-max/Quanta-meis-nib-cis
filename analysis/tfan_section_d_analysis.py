"""Reproduce Section D stability summary for the 2025-11-09 TFan sweep."""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from quanta.parser import load_tfan_results, stability_summary


def main() -> None:
    dataset_path = ROOT / "data" / "t_fan_results_2025-11-09.json"
    dataset = load_tfan_results(dataset_path)
    summary = stability_summary(dataset.entries)

    avg_noise = mean(dataset.spectral_noise())

    print("Transverse Fan Array — Section D Summary")
    print(f"Captured at: {dataset.captured_at:%Y-%m-%d %H:%M:%SZ}")
    print(f"Facility: {dataset.facility}")
    print(f"Protocol version: {dataset.protocol_version}")
    print("---")
    print(f"Stability indices (count={summary['count']}):")
    print(f"  mean = {summary['mean']:.4f}")
    print(f"  min  = {summary['min']:.4f}")
    print(f"  max  = {summary['max']:.4f}")
    print("---")
    print(f"Spectral noise (mean) = {avg_noise:.2f} dB")


if __name__ == "__main__":
    main()
