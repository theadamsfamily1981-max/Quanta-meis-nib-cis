"""Report parameter efficiency of LoRA adapters versus dense fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def compute_lora_efficiency(hidden_dims: Dict[str, int], rank: int = 8) -> Dict[str, float]:
    baseline_params = {name: dim * dim for name, dim in hidden_dims.items()}
    lora_params = {name: 2 * dim * rank for name, dim in hidden_dims.items()}
    baseline_total = sum(baseline_params.values())
    lora_total = sum(lora_params.values())
    reduction = 1.0 - lora_total / baseline_total
    return {
        "baseline_total": float(baseline_total),
        "lora_total": float(lora_total),
        "reduction_fraction": float(reduction),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Where to save the LoRA efficiency JSON")
    parser.add_argument("--rank", type=int, default=8)
    args = parser.parse_args()

    hidden_dims = {"vision": 2048, "audio": 1024, "text": 3072}
    metrics = compute_lora_efficiency(hidden_dims, rank=args.rank)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
