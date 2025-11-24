"""False discovery rate helpers.

The real project contained multiple specialised detectors.  For the purposes of
unit testing we provide well documented, deterministic helpers that cover the
most commonly used routines.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Sequence, Tuple


def _validate_p_values(p_values: Sequence[float]) -> List[float]:
    values = [float(v) for v in p_values]
    if not values:
        raise ValueError("p_values must not be empty")
    for value in values:
        if not 0.0 <= value <= 1.0:
            raise ValueError("p_values must be in the unit interval")
    return values


@dataclass(frozen=True)
class FDRResult:
    """Summary produced by :func:`estimate_fdr_thresholds`."""

    threshold: float
    discoveries: Tuple[int, ...]
    q_values: Tuple[float, ...]


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> FDRResult:
    """Benjamini-Hochberg step-up procedure.

    The function returns the threshold, the indices of rejected hypotheses and
    the per-hypothesis *q-values* (adjusted p-values).
    """

    if alpha <= 0 or alpha > 1:
        raise ValueError("alpha must lie in (0, 1]")
    values = _validate_p_values(p_values)
    m = len(values)
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    thresholds = [(i + 1) * alpha / m for i in range(m)]
    discoveries: List[int] = []
    q_values = [0.0] * m
    candidate_threshold = 0.0
    for rank, ((idx, value), threshold) in enumerate(zip(indexed, thresholds)):
        if value <= threshold:
            candidate_threshold = value
            discoveries.append(idx)
        q_values[idx] = min(value * m / (rank + 1), 1.0)
    q_values = _make_monotone(q_values)
    return FDRResult(candidate_threshold, tuple(sorted(discoveries)), tuple(q_values))


def benjamini_yekutieli(p_values: Sequence[float], alpha: float = 0.05) -> FDRResult:
    """Benjamini-Yekutieli correction for dependent tests."""

    values = _validate_p_values(p_values)
    m = len(values)
    harmonic = sum(1.0 / (i + 1) for i in range(m))
    corrected_alpha = alpha / harmonic
    return benjamini_hochberg(values, corrected_alpha)


def _make_monotone(q_values: List[float]) -> Tuple[float, ...]:
    monotone = list(q_values)
    for i in range(len(monotone) - 2, -1, -1):
        monotone[i] = min(monotone[i], monotone[i + 1])
    return tuple(monotone)


def estimate_fdr_thresholds(
    p_values: Sequence[float],
    *,
    alpha: float = 0.05,
    method: str = "bh",
) -> FDRResult:
    """Wrapper that exposes multiple classical FDR control methods."""

    method = method.lower()
    if method == "bh":
        return benjamini_hochberg(p_values, alpha)
    if method in {"by", "benjamini-yekutieli"}:
        return benjamini_yekutieli(p_values, alpha)
    raise ValueError(f"unknown FDR method: {method}")


def suggest_lr_and_temperature(
    gradients: Sequence[float],
    *,
    max_lr: float = 1.0,
    base_temperature: float = 1.0,
) -> Tuple[float, float]:
    """Return a conservative learning-rate/temperature pair.

    The heuristic mimics a tiny portion of the internal tooling – we compute the
    inter-quartile range of the gradient magnitudes and use it to pick a learning
    rate that keeps the updates numerically stable.  The suggested temperature is
    adjusted so that more volatile gradients (large IQR) result in slightly lower
    temperatures, encouraging the attention distribution to become sharper.
    """

    if max_lr <= 0:
        raise ValueError("max_lr must be positive")
    mags = sorted(abs(float(g)) for g in gradients if not math.isnan(float(g)))
    if not mags:
        return max_lr, base_temperature
    q1 = _percentile(mags, 0.25)
    q3 = _percentile(mags, 0.75)
    iqr = max(q3 - q1, 1e-12)
    lr = min(max_lr, 1.0 / (1.0 + iqr))
    temperature = max(1e-3, base_temperature / (1.0 + iqr))
    return lr, temperature


def _percentile(values: Sequence[float], q: float) -> float:
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0, 1]")
    if not values:
        raise ValueError("values must not be empty")
    pos = q * (len(values) - 1)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return values[int(pos)]
    lower_value = values[lower]
    upper_value = values[upper]
    fraction = pos - lower
    return lower_value + fraction * (upper_value - lower_value)


__all__ = [
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "estimate_fdr_thresholds",
    "suggest_lr_and_temperature",
    "FDRResult",
]
