"""Entrypoint for the GRTES Phase III prototype experiment.

The script stitches together the Φ extrapolation utilities, the category
controller and the neuromorphic hardware simulator.  It can be executed as a
standalone program but is intentionally structured so that tests or notebooks
can import ``run_experiment`` and reuse the orchestration logic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List

from core.cat_controller import CategoryController, CategoryDecision
from core.phi_extrap import PhiExtrapolator
from hardware.energy_profiler import EnergyProfiler
from hardware.neuromorphic_adapter import NeuromorphicAdapter


@dataclass
class ExperimentResult:
    decisions: List[CategoryDecision]
    energy_report: str


def _generate_phi_signal(length: int) -> Iterable[float]:
    """Generate a pseudo-random but smooth Φ signal for demonstration."""

    base = random.uniform(0.1, 0.6)
    for idx in range(length):
        noise = random.uniform(-0.05, 0.05)
        trend = 0.02 * idx / max(length, 1)
        yield max(0.0, min(1.0, base + trend + noise))


def run_experiment(samples: int = 32, operations: int = 512) -> ExperimentResult:
    extrapolator = PhiExtrapolator()
    controller = CategoryController()
    profiler = EnergyProfiler()
    adapter = NeuromorphicAdapter(profiler=profiler)

    decisions: List[CategoryDecision] = []
    for value in _generate_phi_signal(samples):
        extrapolator.observe(value)
        extrapolation = extrapolator.extrapolate(future_steps=3)
        decision = controller.evaluate(extrapolation)
        decisions.append(decision)
        adapter.apply_decision(decision, operations)

    report = adapter.profiler.report().describe()
    return ExperimentResult(decisions=decisions, energy_report=report)


def main() -> None:
    result = run_experiment()
    print("--- GRTES Phase III Prototype ---")
    for idx, decision in enumerate(result.decisions, start=1):
        print(f"t{idx:02d}: {decision.level.value}\t{decision.rationale}")
    print("Energy:", result.energy_report)


if __name__ == "__main__":  # pragma: no cover - manual invocation entrypoint
    main()
