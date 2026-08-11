"""Rayleigh dissipation models for geometric field theory experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class RayleighDissipationFunctional:
    """A quadratic dissipation functional ``R = 1/2 v^T D v``.

    The dissipation tensor ``D`` is assumed to be positive definite.  To keep
    the library lightweight we represent ``D`` by its diagonal elements which
    suffices for many experiments and avoids matrix multiplications.
    """

    damping: torch.Tensor

    def __post_init__(self) -> None:
        if self.damping.ndim != 1:
            raise ValueError("Damping tensor must be one-dimensional")
        if torch.any(self.damping <= 0):
            raise ValueError("Damping coefficients must be positive")

    def quadratic_form(self, velocity: torch.Tensor) -> torch.Tensor:
        """Evaluate the dissipation functional for a velocity vector."""

        if velocity.shape[-1] != self.damping.shape[0]:
            raise ValueError("Velocity dimension does not match damping tensor")
        return 0.5 * (self.damping * velocity ** 2).sum(dim=-1)

    def gradient(self, velocity: torch.Tensor) -> torch.Tensor:
        """Return the gradient of the dissipation functional."""

        if velocity.shape[-1] != self.damping.shape[0]:
            raise ValueError("Velocity dimension does not match damping tensor")
        return self.damping * velocity


__all__: Tuple[str, ...] = ("RayleighDissipationFunctional",)
