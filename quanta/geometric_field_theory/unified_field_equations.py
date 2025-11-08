"""High-level orchestration of the geometric field theory components."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch

try:  # pragma: no cover - optional dependency for persistence calculations
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - allow runtime environments without numpy
    np = None  # type: ignore[assignment]

from .einstein_fields import EinsteinFieldDynamics
from .fisher_rao_metric import FisherRaoMetric
from .persistent_homology_stability import PersistentHomologyAnalyser, PersistenceResult
from .rayleigh_dissipation import RayleighDissipationFunctional


@dataclass
class UnifiedFieldState:
    """Represent the evolving state of the unified field equations."""

    probabilities: torch.Tensor
    velocity: torch.Tensor
    persistence: PersistenceResult | None = None


class UnifiedFieldEquations:
    """Combine geometric, dissipative, and topological computations."""

    def __init__(
        self,
        damping: Iterable[float],
        *,
        max_persistence_dim: int = 1,
        device: torch.device | None = None,
    ) -> None:
        self.metric = FisherRaoMetric()
        damping_tensor = torch.as_tensor(tuple(damping), dtype=torch.float32, device=device)
        self.dissipation = RayleighDissipationFunctional(damping=damping_tensor)
        self.dynamics = EinsteinFieldDynamics(metric=self.metric, dissipation=self.dissipation)
        self.persistence = PersistentHomologyAnalyser(maxdim=max_persistence_dim)

    def step(self, state: UnifiedFieldState, dt: float) -> UnifiedFieldState:
        """Advance the system by a single explicit Euler step."""

        if dt <= 0:
            raise ValueError("Time step must be positive")
        update = self.dynamics.field_update(state.probabilities, state.velocity)
        new_probabilities = self.metric.normalise(state.probabilities + dt * update)
        new_velocity = state.velocity + dt * (-self.dissipation.gradient(state.velocity))
        if np is None:
            raise RuntimeError("numpy is required to advance the unified field equations")
        point_cloud = np.asarray(state.probabilities.detach().cpu(), dtype=float).reshape(-1, 1)
        persistence = self.persistence.compute(point_cloud)
        return UnifiedFieldState(new_probabilities, new_velocity, persistence)

    def diagnostics(self, state: UnifiedFieldState) -> Dict[str, float]:
        """Return lightweight diagnostic scalars."""

        fisher_energy = float(self.metric.quadratic_form(state.probabilities, state.velocity).mean().item())
        dissipation = float(self.dissipation.quadratic_form(state.velocity).mean().item())
        bottleneck = float(state.persistence.bottleneck) if state.persistence else float("nan")
        return {
            "fisher_energy": fisher_energy,
            "dissipation": dissipation,
            "bottleneck": bottleneck,
        }


__all__: Tuple[str, ...] = ("UnifiedFieldEquations", "UnifiedFieldState")
