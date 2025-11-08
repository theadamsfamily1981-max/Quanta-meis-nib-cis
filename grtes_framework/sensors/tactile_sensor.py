"""Tactile sensing primitives for the GR-TES research framework.

This module introduces simple data structures and processing helpers that
can be reused by different experiment runners.  The implementation is kept
framework-agnostic so that it can be imported both by research notebooks
and standalone experiment scripts.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Iterable, List, MutableMapping


@dataclass(frozen=True)
class TactileSample:
    """Immutable container describing a single tactile sensor sample."""

    timestamp: float
    pressure: float
    temperature: float
    vibration: float

    def as_vector(self) -> List[float]:
        """Return the sample values as a dense vector for downstream math."""

        return [self.pressure, self.temperature, self.vibration]


class TactileSensor:
    """Simple tactile sensor abstraction.

    The implementation models a grid-based sensor with three primary
    channels: pressure, temperature and vibration.  The public API focuses
    on reproducibility and deterministic computation, which is important for
    research experiments and offline analysis.
    """

    def __init__(self, sensor_id: str, sampling_rate_hz: float, noise_floor: float = 0.02) -> None:
        if sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if noise_floor < 0:
            raise ValueError("noise_floor cannot be negative")

        self.sensor_id = sensor_id
        self.sampling_rate_hz = sampling_rate_hz
        self.noise_floor = noise_floor
        self._baseline = TactileSample(0.0, 0.0, 0.0, 0.0)

    @property
    def baseline(self) -> TactileSample:
        """Expose the current baseline sample used for normalization."""

        return self._baseline

    def calibrate(self, baseline_samples: Iterable[TactileSample]) -> None:
        """Compute a baseline from raw samples and store it locally.

        The calibration step is usually executed once before a series of
        experiments to remove static offsets (for instance the weight of a
        silicone shell around the sensor).
        """

        samples = list(baseline_samples)
        if not samples:
            raise ValueError("baseline_samples must not be empty")

        self._baseline = TactileSample(
            timestamp=samples[-1].timestamp,
            pressure=fmean(sample.pressure for sample in samples),
            temperature=fmean(sample.temperature for sample in samples),
            vibration=fmean(sample.vibration for sample in samples),
        )

    def record(self, timestamp: float, channel_values: MutableMapping[str, float]) -> TactileSample:
        """Convert a raw channel mapping into a :class:`TactileSample`.

        The helper performs minimal validation to keep downstream code
        focused on experiment logic.
        """

        try:
            pressure = float(channel_values["pressure"])
            temperature = float(channel_values["temperature"])
            vibration = float(channel_values["vibration"])
        except KeyError as error:
            raise KeyError(f"Missing tactile channel: {error.args[0]}") from error

        return TactileSample(timestamp=timestamp, pressure=pressure, temperature=temperature, vibration=vibration)

    def normalize(self, sample: TactileSample) -> List[float]:
        """Normalize a sample using the current baseline."""

        baseline = self._baseline
        return [
            sample.pressure - baseline.pressure,
            sample.temperature - baseline.temperature,
            sample.vibration - baseline.vibration,
        ]
