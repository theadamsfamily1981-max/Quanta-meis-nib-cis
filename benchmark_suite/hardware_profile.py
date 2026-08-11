"""Hardware profiling utilities for Phase III benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping


@dataclass
class HardwareSensorReading:
    """Single point-in-time measurement collected from the hardware stack."""

    metric: str
    value: float
    unit: str

    def to_dict(self) -> Dict[str, object]:
        return {"metric": self.metric, "value": self.value, "unit": self.unit}


@dataclass
class HardwareProfile:
    """Representation of the hardware used during benchmarking."""

    name: str
    accelerator: str
    compute_capability: float
    memory_gb: int
    power_limit_w: int
    sensors: List[HardwareSensorReading] = field(default_factory=list)

    def record_sensor(self, metric: str, value: float, unit: str) -> None:
        """Record a sensor reading for later introspection."""

        self.sensors.append(HardwareSensorReading(metric=metric, value=value, unit=unit))

    def summary(self) -> Mapping[str, object]:
        """Return a summary including the latest sensor data."""

        return {
            "name": self.name,
            "accelerator": self.accelerator,
            "compute_capability": self.compute_capability,
            "memory_gb": self.memory_gb,
            "power_limit_w": self.power_limit_w,
            "sensors": [sensor.to_dict() for sensor in self.sensors],
            "efficiency_score": round(self._efficiency_score(), 3),
        }

    def _efficiency_score(self) -> float:
        """Compute a simple efficiency score based on memory and power."""

        return (self.memory_gb / max(self.power_limit_w, 1)) * self.compute_capability * 10


def default_hardware_profile() -> HardwareProfile:
    """Construct a representative hardware profile for the launch kit."""

    profile = HardwareProfile(
        name="quantum_edge_node",
        accelerator="GRTES-QN200",
        compute_capability=7.2,
        memory_gb=48,
        power_limit_w=350,
    )
    profile.record_sensor(metric="idle_power", value=110.5, unit="w")
    profile.record_sensor(metric="load_power", value=298.0, unit="w")
    profile.record_sensor(metric="temperature", value=67.2, unit="celsius")
    return profile


def merge_profiles(profiles: Iterable[HardwareProfile]) -> Mapping[str, object]:
    """Aggregate hardware summaries for multiple profiles."""

    summaries = [profile.summary() for profile in profiles]
    mean_efficiency = sum(item["efficiency_score"] for item in summaries) / max(len(summaries), 1)
    return {"profiles": summaries, "mean_efficiency": round(mean_efficiency, 3)}
