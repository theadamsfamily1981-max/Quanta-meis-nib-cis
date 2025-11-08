"""Topological Field-Adaptive Network (T-FAN) v0.1
================================================================

This module provides a minimal-yet-functional reference implementation of
an end-to-end pipeline that combines neural feature extractors with
Topological Data Analysis (TDA) insights.  The core model is intentionally
compact so it can be used as a pedagogical baseline or as a quick smoke-test
inside CI environments.

The main entry points are:

```
python -m models.tfan_v0_1 --run-suite
python models/tfan_v0_1.py --export-demo demo.png
```

Both commands will verify that the required optional dependencies are
available before executing.  The script emits synthetic trajectories,
trains a tiny neural field encoder, and measures topological summaries via
Giotto-TDA.  The pipeline is deterministic by design to keep regression
checks stable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


REQUIRED_MODULES: Sequence[str] = ("torch", "gtda", "matplotlib")
SEED = 42


def _ensure_dependencies() -> None:
    """Raise a helpful error if optional runtime dependencies are missing."""

    missing = [pkg for pkg in REQUIRED_MODULES if importlib.util.find_spec(pkg) is None]
    if missing:
        raise ModuleNotFoundError(
            "T-FAN requires optional dependencies that are not installed: "
            + ", ".join(sorted(missing))
        )


def _set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        # numpy is not strictly required, so we tolerate its absence.
        pass


@dataclass
class TFanConfig:
    """Configuration options for the T-FAN demonstration pipeline."""

    input_dim: int = 2
    hidden_dim: int = 32
    persistence_dim: int = 2
    learning_rate: float = 5e-3
    steps: int = 256
    batch_size: int = 64
    device: str = "cpu"
    export_demo: str | None = None
    results_path: Path | None = None


class TFanNetwork:
    """A compact neural field encoder with residual connections."""

    def __init__(self, config: TFanConfig):
        _ensure_dependencies()

        import torch
        import torch.nn as nn

        self.config = config
        self.model = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_dim, 1),
        ).to(config.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    def train_step(self, inputs, targets) -> float:
        import torch

        self.optimizer.zero_grad(set_to_none=True)
        preds = self.model(inputs)
        loss = self.loss_fn(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()


def generate_spiral_dataset(num_samples: int, noise: float = 0.02):
    """Create a simple planar spiral that carries non-trivial topology."""

    import torch

    radii = torch.linspace(0.1, 1.0, num_samples)
    angles = radii * 8 * math.pi
    x = radii * torch.cos(angles)
    y = radii * torch.sin(angles)
    coords = torch.stack([x, y], dim=1)
    coords += noise * torch.randn_like(coords)
    labels = torch.sin(angles).unsqueeze(1)
    return coords, labels


def compute_persistence_diagram(points) -> List[float]:
    """Compute 0D/1D persistence lifetimes from the provided points."""

    from gtda.homology import VietorisRipsPersistence
    import torch

    vr = VietorisRipsPersistence(homology_dimensions=(0, 1))
    diagrams = vr.fit_transform(points.unsqueeze(0).detach().cpu().numpy())
    lifetimes: List[float] = []
    for diagram in diagrams:
        if diagram.size == 0:
            continue
        lifetimes.extend((diagram[:, 1] - diagram[:, 0]).tolist())
    return lifetimes


def run_demo(config: TFanConfig) -> dict:
    """Train the T-FAN encoder and compute summary statistics."""

    _ensure_dependencies()
    _set_seed()

    import matplotlib.pyplot as plt
    import torch

    device = torch.device(config.device)
    model = TFanNetwork(config)

    coords, labels = generate_spiral_dataset(config.steps)
    coords = coords.to(device)
    labels = labels.to(device)

    losses: List[float] = []
    for _ in range(config.steps):
        idx = torch.randperm(coords.size(0))[: config.batch_size]
        batch_inputs = coords[idx]
        batch_targets = labels[idx]
        loss = model.train_step(batch_inputs, batch_targets)
        losses.append(loss)

    persistence_lifetimes = compute_persistence_diagram(coords.cpu())

    results = {
        "loss_mean": statistics.fmean(losses),
        "loss_std": statistics.pstdev(losses),
        "persistence_lifetimes": persistence_lifetimes,
        "persistence_summary": {
            "count": len(persistence_lifetimes),
            "max": max(persistence_lifetimes) if persistence_lifetimes else 0.0,
            "mean": statistics.fmean(persistence_lifetimes)
            if persistence_lifetimes
            else 0.0,
        },
    }

    if config.export_demo:
        fig, ax = plt.subplots(figsize=(6, 6))
        coords_np = coords.detach().cpu().numpy()
        scatter = ax.scatter(coords_np[:, 0], coords_np[:, 1], c=labels.cpu().numpy(), cmap="viridis")
        ax.set_title("T-FAN v0.1 synthetic spiral dataset")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(scatter, ax=ax, label="target signal")
        export_path = Path(config.export_demo)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(export_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    if config.results_path:
        config.results_path.parent.mkdir(parents=True, exist_ok=True)
        config.results_path.write_text(json.dumps(results, indent=2, sort_keys=True))

    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the T-FAN v0.1 demonstration suite.")
    parser.add_argument("--run-suite", action="store_true", help="Execute the smoke-test training loop")
    parser.add_argument(
        "--export-demo",
        metavar="PATH",
        help="Optional path to export a PNG visualisation of the synthetic dataset.",
    )
    parser.add_argument(
        "--results",
        metavar="PATH",
        help="Optional JSON file where metrics/results will be stored.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to use (default: cpu).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=256,
        help="Number of gradient steps for the demonstration training loop.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size for stochastic updates.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=32,
        help="Width of the hidden representation inside the encoder.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.run_suite and not args.export_demo:
        print("No action requested. Use --run-suite to execute the demo.")
        return 0

    config = TFanConfig(
        hidden_dim=args.hidden_dim,
        steps=args.steps,
        batch_size=args.batch_size,
        device=args.device,
        export_demo=args.export_demo,
        results_path=Path(args.results) if args.results else None,
    )

    try:
        results = run_demo(config)
    except ModuleNotFoundError as exc:  # pragma: no cover - ensures friendly CLI output
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if config.results_path:
        print(f"Results stored at {config.results_path}")
    else:
        print(json.dumps(results, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    sys.exit(main())
