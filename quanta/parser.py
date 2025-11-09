"""Utilities for parsing Transverse Fan Array empirical datasets.

These helpers are referenced in Section D of the LaTeX supplement and provide
strong typing plus convenience aggregations for the 2025-11-09 sweep.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class FanResult:
    """Single row of the transverse fan array sweep."""

    fan_id: str
    timestamp: datetime
    ambient_temperature_c: float
    ambient_humidity: float
    stability_index: float
    flow_rate_cms: float
    spectral_noise_db: float
    runtime_minutes: int
    phase_shift_deg: float

    @classmethod
    def from_json(cls, payload: dict) -> "FanResult":
        """Create a :class:`FanResult` from a JSON dictionary."""

        return cls(
            fan_id=payload["fan_id"],
            timestamp=datetime.fromisoformat(payload["timestamp"].replace("Z", "+00:00")),
            ambient_temperature_c=float(payload["ambient_temperature_c"]),
            ambient_humidity=float(payload["ambient_humidity"]),
            stability_index=float(payload["stability_index"]),
            flow_rate_cms=float(payload["flow_rate_cms"]),
            spectral_noise_db=float(payload["spectral_noise_db"]),
            runtime_minutes=int(payload["runtime_minutes"]),
            phase_shift_deg=float(payload["phase_shift_deg"]),
        )


@dataclass(frozen=True)
class TFanDataset:
    """Container for the full transverse fan dataset."""

    captured_at: datetime
    protocol_version: str
    facility: str
    notes: str
    entries: Sequence[FanResult]

    def stability_indices(self) -> List[float]:
        """Return the list of stability index scores across all entries."""

        return [entry.stability_index for entry in self.entries]

    def spectral_noise(self) -> List[float]:
        """Return the list of spectral noise estimates in dB."""

        return [entry.spectral_noise_db for entry in self.entries]


def _coerce_entries(raw_entries: Iterable[dict]) -> List[FanResult]:
    return [FanResult.from_json(entry) for entry in raw_entries]


def load_tfan_results(path: Path | str) -> TFanDataset:
    """Load a transverse fan dataset from the canonical JSON export."""

    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    entries = _coerce_entries(payload["entries"])

    return TFanDataset(
        captured_at=datetime.fromisoformat(payload["captured_at"].replace("Z", "+00:00")),
        protocol_version=payload["protocol_version"],
        facility=payload["facility"],
        notes=payload.get("notes", ""),
        entries=entries,
    )


def stability_summary(entries: Sequence[FanResult]) -> dict:
    """Compute high-level summary statistics for stability indices."""

    if not entries:
        raise ValueError("entries must not be empty")

    indices = [entry.stability_index for entry in entries]
    return {
        "count": len(indices),
        "mean": mean(indices),
        "min": min(indices),
        "max": max(indices),
    }


__all__ = [
    "FanResult",
    "TFanDataset",
    "load_tfan_results",
    "stability_summary",
]
