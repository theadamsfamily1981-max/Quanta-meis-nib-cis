"""Core tactile integration routines used during Phase III experiments."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import fmean
from typing import List, Mapping, MutableSequence, Sequence

from grtes_framework.sensors.tactile_sensor import TactileSample, TactileSensor


@dataclass(frozen=True)
class TactileTestResult:
    """Structured representation of a single tactile experiment outcome."""

    test_id: str
    label: str
    passed: bool
    peak_pressure: float
    avg_temperature: float
    vibration_rms: float

    def as_dict(self) -> Mapping[str, object]:
        """Serialize the result for JSON output."""

        return {
            "test_id": self.test_id,
            "label": self.label,
            "passed": self.passed,
            "peak_pressure": round(self.peak_pressure, 4),
            "avg_temperature": round(self.avg_temperature, 4),
            "vibration_rms": round(self.vibration_rms, 4),
        }


class TactileIntegrationEngine:
    """Combine tactile sensor data with heuristic thresholds.

    The engine is intentionally lightweight—heavyweight modelling happens in the
    downstream analytics stack, whereas here we want deterministic baselines for
    regression-style testing.
    """

    def __init__(self, sensor: TactileSensor, thresholds: Mapping[str, float] | None = None) -> None:
        default_thresholds = {
            "pressure_min": 0.15,
            "pressure_max": 1.8,
            "temperature_max": 0.9,
            "vibration_max": 0.55,
        }
        if thresholds is None:
            thresholds = default_thresholds
        else:
            merged = default_thresholds.copy()
            merged.update(thresholds)
            thresholds = merged

        self.sensor = sensor
        self.thresholds = thresholds

    def evaluate_contact(self, samples: Sequence[TactileSample]) -> Mapping[str, float | bool]:
        """Evaluate a list of tactile samples and compute aggregate metrics."""

        if not samples:
            raise ValueError("samples must not be empty")

        normalized: List[List[float]] = [self.sensor.normalize(sample) for sample in samples]
        peak_pressure = max(vec[0] for vec in normalized)
        avg_temperature = fmean(vec[1] for vec in normalized)
        vibration_rms = sqrt(fmean(vec[2] ** 2 for vec in normalized))

        thresholds = self.thresholds
        passed = (
            thresholds["pressure_min"] <= peak_pressure <= thresholds["pressure_max"]
            and avg_temperature <= thresholds["temperature_max"]
            and vibration_rms <= thresholds["vibration_max"]
        )

        return {
            "peak_pressure": peak_pressure,
            "avg_temperature": avg_temperature,
            "vibration_rms": vibration_rms,
            "passed": passed,
        }

    def run_test_matrix(self, matrix: Sequence[Mapping[str, object]]) -> List[TactileTestResult]:
        """Execute a matrix of tactile tests and return structured results."""

        results: MutableSequence[TactileTestResult] = []
        for entry in matrix:
            test_id = str(entry["id"])
            label = str(entry.get("label", ""))
            samples: Sequence[TactileSample] = entry["samples"]  # type: ignore[assignment]
            metrics = self.evaluate_contact(samples)
            results.append(
                TactileTestResult(
                    test_id=test_id,
                    label=label,
                    passed=bool(metrics["passed"]),
                    peak_pressure=float(metrics["peak_pressure"]),
                    avg_temperature=float(metrics["avg_temperature"]),
                    vibration_rms=float(metrics["vibration_rms"]),
                )
            )
        return list(results)
