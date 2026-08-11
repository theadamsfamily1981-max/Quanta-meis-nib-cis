"""Phase III tactile integration experiment runner.

The script synthesises a deterministic set of tactile contacts covering
forty experimental setups.  The results are intended for regression testing
of the tactile sensing stack and double as a lightweight documentation of
expected behaviour for each scenario.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from random import Random
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grtes_framework.core.tactile_integration import TactileIntegrationEngine
from grtes_framework.sensors.tactile_sensor import TactileSample, TactileSensor

RESULT_PATH = Path("results/phaseIII_tactile_results.json")


def _create_sensor() -> TactileSensor:
    sensor = TactileSensor("phaseIII-touch-array", sampling_rate_hz=120.0, noise_floor=0.015)
    rng = Random(4242)
    baseline_samples = [
        TactileSample(
            timestamp=index / sensor.sampling_rate_hz,
            pressure=0.08 + rng.uniform(-0.005, 0.005),
            temperature=0.05 + rng.uniform(-0.004, 0.004),
            vibration=0.02 + rng.uniform(-0.003, 0.003),
        )
        for index in range(30)
    ]
    sensor.calibrate(baseline_samples)
    return sensor


def _generate_samples(
    sensor: TactileSensor,
    seed: int,
    pressure_offset: float,
    temperature_offset: float,
    vibration_offset: float,
    sample_count: int = 64,
) -> List[TactileSample]:
    rng = Random(seed)
    samples: List[TactileSample] = []
    for index in range(sample_count):
        timestamp = index / sensor.sampling_rate_hz
        ramp = min(1.0, index / (sample_count / 3.0))
        pressure = sensor.baseline.pressure + max(
            0.0, pressure_offset * ramp + rng.uniform(-0.05, 0.05) * max(pressure_offset, 0.05)
        )
        temperature = sensor.baseline.temperature + max(
            0.0, temperature_offset * ramp + rng.uniform(-0.03, 0.03) * max(temperature_offset, 0.03)
        )
        vibration = sensor.baseline.vibration + max(
            0.0, vibration_offset * ramp + rng.uniform(-0.04, 0.04) * max(vibration_offset, 0.04)
        )
        samples.append(
            TactileSample(
                timestamp=timestamp,
                pressure=pressure,
                temperature=temperature,
                vibration=vibration,
            )
        )
    return samples


BASE_SCENARIOS: Sequence[Dict[str, float | str]] = [
    {"name": "Soft foam grip", "pressure": 0.32, "temperature": 0.18, "vibration": 0.12},
    {"name": "Cable harness pinch", "pressure": 0.54, "temperature": 0.22, "vibration": 0.26},
    {"name": "Rigid block press", "pressure": 1.35, "temperature": 0.24, "vibration": 0.18},
    {"name": "Silicone membrane sweep", "pressure": 0.28, "temperature": 0.16, "vibration": 0.34},
    {"name": "Thermal drift monitor", "pressure": 0.42, "temperature": 0.38, "vibration": 0.22},
    {"name": "Microtexture mapping", "pressure": 0.36, "temperature": 0.28, "vibration": 0.48},
    {"name": "Valve rotation support", "pressure": 0.58, "temperature": 0.21, "vibration": 0.32},
    {"name": "Compliance rating", "pressure": 0.73, "temperature": 0.19, "vibration": 0.29},
    {"name": "Impact damping check", "pressure": 0.49, "temperature": 0.17, "vibration": 0.52},
    {"name": "Soft grasp release", "pressure": 0.31, "temperature": 0.20, "vibration": 0.14},
]

PRESSURE_VARIANTS = [0.0, 0.18, -0.22, 2.1]
TEMPERATURE_VARIANTS = [0.0, 0.05, 0.02, 0.12]
VIBRATION_VARIANTS = [0.0, 0.04, 0.02, 0.12]


def build_test_matrix(sensor: TactileSensor) -> List[Dict[str, object]]:
    matrix: List[Dict[str, object]] = []
    for offset, test_id in enumerate(range(331, 371)):
        scenario = BASE_SCENARIOS[offset % len(BASE_SCENARIOS)]
        variant_index = offset // len(BASE_SCENARIOS)
        label = f"{scenario['name']} – variant {variant_index + 1}"

        pressure = float(scenario["pressure"]) + PRESSURE_VARIANTS[variant_index]
        temperature = float(scenario["temperature"]) + TEMPERATURE_VARIANTS[variant_index]
        vibration = float(scenario["vibration"]) + VIBRATION_VARIANTS[variant_index]

        samples = _generate_samples(
            sensor=sensor,
            seed=10_000 + test_id,
            pressure_offset=pressure,
            temperature_offset=temperature,
            vibration_offset=vibration,
        )

        matrix.append({"id": f"Exp-{test_id}", "label": label, "samples": samples})

    return matrix


def run_suite(output_path: Path = RESULT_PATH) -> Dict[str, object]:
    sensor = _create_sensor()
    engine = TactileIntegrationEngine(sensor)
    matrix = build_test_matrix(sensor)
    results = engine.run_test_matrix(matrix)

    payload = {
        "suite": "Phase III tactile integration",
        "sensor": sensor.sensor_id,
        "test_count": len(results),
        "passed": sum(1 for result in results if result.passed),
        "failed": sum(1 for result in results if not result.passed),
        "results": [result.as_dict() for result in results],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    payload = run_suite()
    print(json.dumps({"summary": {"passed": payload["passed"], "failed": payload["failed"]}}, indent=2))


if __name__ == "__main__":
    main()
