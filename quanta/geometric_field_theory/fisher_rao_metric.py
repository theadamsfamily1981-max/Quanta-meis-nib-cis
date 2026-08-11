"""Utilities for working with the Fisher-Rao information metric.

The implementation is intentionally lightweight.  It only depends on ``numpy``
and ``torch`` and therefore integrates well with a variety of optimisation
routines.  The routines are built to operate on probability simplices and thus
normalise their inputs when necessary.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import acos
from typing import Iterable, Tuple

import torch

try:  # pragma: no cover - optional dependency for distance computation
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - allow runtime environments without numpy
    np = None  # type: ignore[assignment]


@dataclass
class FisherRaoMetric:
    """Compute the Fisher-Rao metric for discrete probability models.

    The Fisher-Rao metric for a discrete distribution ``p`` with tangent vector
    ``v`` is :math:`\langle v, v \rangle_p = \sum_i v_i^2 / p_i`.  We expose two
    convenience methods: :meth:`normalise` to move an arbitrary tensor onto the
    probability simplex and :meth:`quadratic_form` to evaluate the quadratic
    form defined by the metric.
    """

    eps: float = 1e-12

    def normalise(self, values: torch.Tensor) -> torch.Tensor:
        """Project ``values`` onto the probability simplex.

        Parameters
        ----------
        values:
            A tensor containing (possibly unnormalised) non-negative values.

        Returns
        -------
        torch.Tensor
            The normalised tensor whose entries sum to one along the last axis.
        """

        if values.ndim == 0:
            raise ValueError("Values must have at least one dimension")
        if torch.any(values < 0):
            raise ValueError("Values must be non-negative")
        total = values.sum(dim=-1, keepdim=True)
        if torch.any(total <= 0):
            raise ValueError("Values must sum to a positive quantity")
        return values / total

    def quadratic_form(self, probabilities: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        """Evaluate the Fisher-Rao quadratic form for a tangent vector.

        Both ``probabilities`` and ``tangent`` are broadcast so they can accept
        batched inputs.  The function clamps the denominator to avoid numerical
        issues when probabilities approach zero.
        """

        probs = self.normalise(probabilities)
        if probs.shape != tangent.shape:
            try:
                tangent = torch.broadcast_to(tangent, probs.shape)
            except RuntimeError as exc:  # pragma: no cover - informative message
                raise ValueError("Tangent could not be broadcast to probabilities") from exc
        denom = torch.clamp(probs, min=self.eps)
        return (tangent ** 2 / denom).sum(dim=-1)

    def metric_tensor(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Return the diagonal Fisher information matrix for discrete models."""

        probs = self.normalise(probabilities)
        denom = torch.clamp(probs, min=self.eps)
        return torch.diag_embed(1.0 / denom)

    @staticmethod
    def fisher_distance(p: Iterable[float], q: Iterable[float]) -> float:
        """Compute the Fisher-Rao geodesic distance between two distributions."""

        p_tuple, q_tuple = tuple(float(x) for x in p), tuple(float(x) for x in q)
        if len(p_tuple) == 0 or len(q_tuple) == 0:
            raise ValueError("Input distributions must be non-empty")
        if len(p_tuple) != len(q_tuple):
            raise ValueError("Distributions must have the same length")
        if any(x < 0 for x in p_tuple) or any(x < 0 for x in q_tuple):
            raise ValueError("Probabilities must be non-negative")
        total_p = sum(p_tuple)
        total_q = sum(q_tuple)
        if total_p <= 0 or total_q <= 0:
            raise ValueError("Probabilities must sum to a positive value")
        p_norm = [x / total_p for x in p_tuple]
        q_norm = [x / total_q for x in q_tuple]
        inner = sum((pp * qq) ** 0.5 for pp, qq in zip(p_norm, q_norm))
        inner = max(min(inner, 1.0), -1.0)
        return float(2 * acos(inner))


__all__: Tuple[str, ...] = ("FisherRaoMetric",)
