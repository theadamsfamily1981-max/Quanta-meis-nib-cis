"""Loss helpers for the TFAN pipeline.

The implementations rely solely on the Python standard library so tests can run
in extremely minimal CI environments.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass
class LossResult:
    """Container for scalar loss values.

    Each field corresponds to a tracked objective in the historical TFAN code
    base.  The dataclass makes it convenient to serialize metrics or iterate
    over them in the CLI entrypoint.
    """

    landmark_attention: float
    alpha_probe: float
    jt_fan: float

    def as_dict(self) -> dict[str, float]:
        """Return a plain mapping for logging or tabular output."""

        return {
            "landmark_attention": self.landmark_attention,
            "alpha_probe": self.alpha_probe,
            "jt_fan": self.jt_fan,
        }


def _to_list(values: Iterable[float]) -> List[float]:
    if isinstance(values, list):
        return values
    return list(values)


def _normalize(values: Iterable[float]) -> List[float]:
    arr = _to_list(values)
    if not arr:
        return []
    min_val = min(arr)
    arr = [v - min_val for v in arr]
    max_val = max(arr)
    if max_val > 0:
        arr = [v / max_val for v in arr]
    return arr


def landmark_attention_loss(embeddings: Sequence[Sequence[float]]) -> float:
    """Compute a simplified landmark attention loss.

    The routine penalises dispersion of a landmark token with a covariance
    proxy.  While simplified, the shape closely mirrors the original
    implementation and keeps tests stable.
    """

    rows = len(embeddings)
    if rows == 0:
        return 0.0
    cols = len(embeddings[0]) if embeddings[0] else 0
    means = [sum(embeddings[r][c] for r in range(rows)) / rows for c in range(cols)]
    total = 0.0
    for row in embeddings:
        for c, value in enumerate(row):
            diff = value - means[c]
            total += diff * diff
    denom = max(rows - 1, 1)
    return total / denom


def alpha_probe_loss(activations: Iterable[float], targets: Iterable[float]) -> float:
    """Return a surrogate α-probe loss.

    We use a cosine-distance objective that is inexpensive yet demonstrates how
    the previous α-probe interface can be preserved.
    """

    activations = _normalize(activations)
    targets = _normalize(targets)
    numerator = sum(a * b for a, b in zip(activations, targets))
    norm_act = math.sqrt(sum(a * a for a in activations))
    norm_tgt = math.sqrt(sum(b * b for b in targets))
    denom = norm_act * norm_tgt
    cosine = 1.0 if denom == 0 else numerator / denom
    return float(1 - cosine)


def jt_fan_loss(boundary_energy: Iterable[float], kappa: float = 0.3) -> float:
    """A compact JT-FAN objective following the topology/thermodynamics theme."""

    energy = _to_list(boundary_energy)
    if not energy:
        return 0.0
    mean_energy = sum(energy) / len(energy)
    total = 0.0
    for value in energy:
        quadratic = (value - mean_energy) ** 2
        modulation = 1 + kappa * math.sin(value)
        total += quadratic * modulation
    return total / len(energy)


def compute_losses(rng: random.Random, *, token_dim: int = 8) -> LossResult:
    """Sample toy tensors and evaluate the historical TFAN losses."""

    embeddings = [
        [rng.gauss(0, 1) for _ in range(token_dim)]
        for _ in range(32)
    ]
    activations = [rng.gauss(0, 1) for _ in range(token_dim)]
    targets = [rng.gauss(0, 1) for _ in range(token_dim)]
    boundary_energy = [rng.gauss(0, 1) for _ in range(token_dim)]

    return LossResult(
        landmark_attention=landmark_attention_loss(embeddings),
        alpha_probe=alpha_probe_loss(activations, targets),
        jt_fan=jt_fan_loss(boundary_energy),
    )
