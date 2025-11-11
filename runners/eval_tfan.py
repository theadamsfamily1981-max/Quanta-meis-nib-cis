"""Evaluation entry-point for the T-FAN MVP."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tff.topo_regularizer import TopologyRegularizer
from udk.safety_monitor import SafetyMonitor


def evaluate(metrics_path: Path) -> dict:
    data = json.loads(metrics_path.read_text())
    fused_value = data.get("curvature_ratio", 0.0)
    regularizer = TopologyRegularizer()
    curvature_ratio = max(fused_value, 0.0)
    monitor = SafetyMonitor(threshold=0.0)
    safety = monitor.evaluate([data.get("safety_TP", 0.0)], [1.0])
    return {
        "curvature_ratio": curvature_ratio,
        "safety_TP": safety["TP"],
        "FP": data.get("FP", 0.0),
        "speedup_2048": data.get("speedup_2048", 0.0),
        "lora_reduction": data.get("lora_reduction", 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate stored T-FAN metrics")
    parser.add_argument("metrics", type=Path, help="Path to metrics JSON produced by training")
    args = parser.parse_args()

    results = evaluate(args.metrics)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
