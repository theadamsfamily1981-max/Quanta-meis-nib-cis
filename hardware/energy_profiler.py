"""Energy profiling utilities for the neuromorphic hardware simulator."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict

from core.cat_controller import CategoryLevel


@dataclass
class EnergyReport:
    """Aggregated energy statistics for a simulation run."""

    total_energy: float
    per_category: Dict[CategoryLevel, float]

    def describe(self) -> str:
        segments = [f"total={self.total_energy:.3f} pJ"]
        for level, energy in sorted(self.per_category.items(), key=lambda item: item[0].value):
            segments.append(f"{level.value}={energy:.3f} pJ")
        return ", ".join(segments)


class EnergyProfiler:
    """Record energy usage across categories."""

    def __init__(self) -> None:
        self._energy = defaultdict(float)

    def record(self, level: CategoryLevel, energy: float) -> None:
        if energy < 0:
            raise ValueError("energy must be non-negative")
        self._energy[level] += energy

    def reset(self) -> None:
        self._energy.clear()

    def report(self) -> EnergyReport:
        total = sum(self._energy.values())
        return EnergyReport(total_energy=total, per_category=dict(self._energy))


__all__ = ["EnergyProfiler", "EnergyReport"]
