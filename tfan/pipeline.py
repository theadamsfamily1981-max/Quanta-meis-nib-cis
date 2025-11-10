"""High-level pipeline orchestrating TFAN losses and plotting."""
from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Tuple

from . import plotting
from .losses import LossResult, compute_losses


class StructuredCurriculum:
    """Simple generator for a curriculum of structured vs scrambled tasks."""

    def __init__(self, *, n_steps: int = 20, scramble_ratio: float = 0.35):
        self.n_steps = n_steps
        self.scramble_ratio = scramble_ratio

    def generate(self, rng: random.Random) -> Tuple[List[float], List[float]]:
        """Return structured and scrambled accuracy curves."""

        if self.n_steps <= 1:
            return [0.5], [0.5 - self.scramble_ratio]

        step = (0.95 - 0.4) / (self.n_steps - 1)
        base_curve = [0.4 + step * idx for idx in range(self.n_steps)]
        structured: List[float] = []
        scrambled: List[float] = []
        for base in base_curve:
            noisy = max(0.0, min(1.0, base + rng.gauss(0, 0.03)))
            structured.append(noisy)
            penalty = self.scramble_ratio * rng.random()
            scrambled.append(max(0.0, noisy - penalty))

        return structured, scrambled


def run_pipeline(
    *,
    seed: int = 7,
    token_dim: int = 8,
    k_values: Iterable[int] | None = None,
    save_plots: Path | None = None,
) -> LossResult:
    """Evaluate TFAN losses and optionally persist diagnostic plots."""

    rng = random.Random(seed)
    losses = compute_losses(rng, token_dim=token_dim)

    if save_plots is not None:
        save_plots = Path(save_plots)
        save_plots.mkdir(parents=True, exist_ok=True)

        metrics = asdict(losses)
        if metrics:
            step = (1.0 - 0.1) / max(len(metrics) - 1, 1)
            dissipation = [0.1 + idx * step for idx in range(len(metrics))]
        else:
            dissipation = [0.1]
        plotting.save_pareto_front_plot(metrics, dissipation, save_plots)

        if k_values is None:
            k_values = range(1, 6)
        k_list = list(k_values)
        throughput = [rng.uniform(0.5, 1.5) for _ in k_list]
        plotting.save_throughput_plot(k_list, throughput, save_plots)

        curriculum = StructuredCurriculum()
        structured, scrambled = curriculum.generate(rng)
        plotting.save_nfl_curriculum_plot(structured, scrambled, save_plots)

    return losses
