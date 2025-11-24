"""Selective self-attention helpers with temperature scaling.

The real project utilises PyTorch; the re-implementation below keeps the
behaviour but operates purely on Python sequences so it can run without external
runtime dependencies.  Only the pieces that are required in the unit tests are
implemented.
"""
from __future__ import annotations

from typing import List, Sequence
import math


def _normalise_axis(scores: Sequence[float]) -> List[float]:
    if not scores:
        raise ValueError("scores must not be empty")
    if any(math.isnan(v) for v in scores):
        raise ValueError("scores must not contain NaNs")
    max_score = max(scores)
    exp_scores = [math.exp(v - max_score) for v in scores]
    total = sum(exp_scores)
    return [v / total for v in exp_scores]


def temperature_scaled_softmax(scores: Sequence[float], temperature: float = 1.0) -> List[float]:
    """Softmax with temperature scaling.

    ``temperature`` behaves identically to the common deep learning
    interpretation: values > 1.0 flatten the distribution, values < 1.0 sharpen
    it.  ``temperature`` must be strictly positive.
    """

    if temperature <= 0:
        raise ValueError("temperature must be strictly positive")
    scaled = [v / temperature for v in scores]
    return _normalise_axis(scaled)


def selective_self_attention(
    scores: Sequence[Sequence[float]],
    *,
    temperature: float = 1.0,
    mask: Sequence[Sequence[bool]] | None = None,
) -> List[List[float]]:
    """Compute self-attention weights with optional masking.

    ``scores`` is interpreted as a square matrix.  When ``mask`` is provided it
    must have the same shape and ``False`` entries denote positions that should
    be ignored.  The function returns a list of probability distributions, one
    per query.
    """

    if not scores:
        raise ValueError("scores must not be empty")
    n = len(scores)
    if any(len(row) != n for row in scores):
        raise ValueError("scores must describe a square matrix")
    if mask is not None and (len(mask) != n or any(len(row) != n for row in mask)):
        raise ValueError("mask must match the shape of scores")

    output: List[List[float]] = []
    for i, row in enumerate(scores):
        if mask is None:
            filtered = row
        else:
            filtered = [value if mask[i][j] else float("-inf") for j, value in enumerate(row)]
            if all(not flag for flag in mask[i]):
                raise ValueError("mask row cannot mask out every position")
        output.append(temperature_scaled_softmax(filtered, temperature))
    return output


__all__ = ["selective_self_attention", "temperature_scaled_softmax"]
