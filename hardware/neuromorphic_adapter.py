"""Neuromorphic hardware adapter simulation for GRTES Phase III.

The adapter translates high-level category decisions into a stream of events
that approximate how a neuromorphic accelerator might respond.  The simulator
is intentionally lightweight but strives to model latency and energy usage so
that experiments can produce meaningful metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from .energy_profiler import EnergyProfiler
from core.cat_controller import CategoryDecision, CategoryLevel


@dataclass
class HardwareEvent:
    """Represents a single hardware configuration update."""

    timestamp: float
    category: CategoryLevel
    latency: float
    energy_cost: float


class NeuromorphicAdapter:
    """Simulate how the hardware reacts to category decisions."""

    # Nominal energy cost per category in picojoules per operation.  These
    # values were derived from the Phase II hardware validation logs.
    CATEGORY_ENERGY_COSTS: Dict[CategoryLevel, float] = {
        CategoryLevel.LOW: 1.0,
        CategoryLevel.MEDIUM: 2.4,
        CategoryLevel.HIGH: 4.6,
    }

    CATEGORY_LATENCIES: Dict[CategoryLevel, float] = {
        CategoryLevel.LOW: 0.015,
        CategoryLevel.MEDIUM: 0.010,
        CategoryLevel.HIGH: 0.004,
    }

    def __init__(self, profiler: EnergyProfiler | None = None) -> None:
        self._profiler = profiler or EnergyProfiler()
        self._clock = 0.0
        self._events: List[HardwareEvent] = []

    @property
    def events(self) -> List[HardwareEvent]:
        """Return the chronological list of emitted events."""

        return list(self._events)

    @property
    def profiler(self) -> EnergyProfiler:
        return self._profiler

    def apply_decision(self, decision: CategoryDecision, operations: int) -> HardwareEvent:
        """Generate the hardware response for ``decision``.

        Parameters
        ----------
        decision:
            The :class:`CategoryDecision` chosen by the controller.
        operations:
            Number of synaptic operations to execute under this configuration.
        """

        if operations < 0:
            raise ValueError("operations must be non-negative")

        energy_cost = self.CATEGORY_ENERGY_COSTS[decision.level] * operations
        latency = self.CATEGORY_LATENCIES[decision.level] * max(operations, 1)
        self._clock += latency

        event = HardwareEvent(
            timestamp=self._clock,
            category=decision.level,
            latency=latency,
            energy_cost=energy_cost,
        )
        self._events.append(event)
        self._profiler.record(decision.level, energy_cost)
        return event

    def replay(self, decisions: Iterable[CategoryDecision], operations: int) -> List[HardwareEvent]:
        """Convenience helper to apply ``operations`` for every decision."""

        return [self.apply_decision(decision, operations) for decision in decisions]

    def reset(self) -> None:
        """Clear the adapter state."""

        self._clock = 0.0
        self._events.clear()
        self._profiler.reset()


__all__ = ["NeuromorphicAdapter", "HardwareEvent"]
