"""Minimal representation of Einstein-like field dynamics for learning systems."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .fisher_rao_metric import FisherRaoMetric
from .rayleigh_dissipation import RayleighDissipationFunctional


@dataclass
class EinsteinFieldDynamics:
    """Couple Fisher information geometry with Rayleigh dissipation."""

    metric: FisherRaoMetric
    dissipation: RayleighDissipationFunctional

    def curvature_scalar(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Return a scalar curvature surrogate for discrete systems.

        We approximate the scalar curvature by the trace of the metric tensor, a
        coarse but informative quantity for toy experiments.
        """

        tensor = self.metric.metric_tensor(probabilities)
        return torch.diagonal(tensor, dim1=-2, dim2=-1).sum(dim=-1)

    def field_update(self, probabilities: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        """Compute a field update combining curvature and dissipation."""

        curvature = self.curvature_scalar(probabilities).unsqueeze(-1)
        damping = self.dissipation.gradient(velocity)
        return -curvature * self.metric.normalise(probabilities) - damping


__all__: Tuple[str, ...] = ("EinsteinFieldDynamics",)
