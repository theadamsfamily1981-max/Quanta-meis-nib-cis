import math

import pytest

torch = pytest.importorskip("torch", reason="torch required for geometric field theory tests")

from quanta.geometric_field_theory.einstein_fields import EinsteinFieldDynamics
from quanta.geometric_field_theory.fisher_rao_metric import FisherRaoMetric
from quanta.geometric_field_theory.rayleigh_dissipation import RayleighDissipationFunctional
from quanta.geometric_field_theory.unified_field_equations import (
    UnifiedFieldEquations,
    UnifiedFieldState,
)


def test_fisher_rao_metric_distance_and_quadratic_form():
    metric = FisherRaoMetric()
    probabilities = torch.tensor([[0.2, 0.3, 0.5]])
    tangent = torch.tensor([[0.1, -0.1, 0.0]])
    quadratic = metric.quadratic_form(probabilities, tangent)
    assert torch.isclose(quadratic, torch.tensor([0.08333333]), atol=1e-5).all()
    distance = metric.fisher_distance([0.5, 0.5], [0.25, 0.75])
    assert math.isclose(distance, 0.910, rel_tol=1e-3)


def test_rayleigh_dissipation_gradient_matches_quadratic_form():
    damping = torch.tensor([0.5, 0.25])
    functional = RayleighDissipationFunctional(damping=damping)
    velocity = torch.tensor([[1.0, -2.0]])
    grad = functional.gradient(velocity)
    quadratic = functional.quadratic_form(velocity)
    assert torch.allclose(grad, torch.tensor([[0.5, -0.5]]))
    assert torch.isclose(quadratic, torch.tensor([1.25]), atol=1e-5).all()


def test_einstein_field_update_couples_curvature_and_dissipation():
    metric = FisherRaoMetric()
    damping = torch.tensor([0.5, 0.5, 0.5])
    dissipation = RayleighDissipationFunctional(damping=damping)
    dynamics = EinsteinFieldDynamics(metric=metric, dissipation=dissipation)
    probabilities = torch.tensor([[0.3, 0.3, 0.4]], requires_grad=False)
    velocity = torch.tensor([[0.1, 0.0, -0.1]])
    update = dynamics.field_update(probabilities, velocity)
    assert update.shape == probabilities.shape
    assert torch.all(update <= 0.0)


def test_unified_field_equations_step_and_diagnostics():
    pytest.importorskip("ripser", reason="ripser required for persistence")
    pytest.importorskip("numpy", reason="numpy required for persistence")
    damping = [0.5, 0.5, 0.5]
    equations = UnifiedFieldEquations(damping, max_persistence_dim=0)
    state = UnifiedFieldState(
        probabilities=torch.tensor([0.2, 0.3, 0.5]),
        velocity=torch.tensor([0.05, -0.05, 0.0]),
    )
    new_state = equations.step(state, dt=0.1)
    diagnostics = equations.diagnostics(new_state)
    assert set(diagnostics.keys()) == {"fisher_energy", "dissipation", "bottleneck"}
    assert diagnostics["fisher_energy"] >= 0
    assert diagnostics["dissipation"] >= 0
    assert math.isfinite(diagnostics["bottleneck"]) or math.isnan(diagnostics["bottleneck"])
