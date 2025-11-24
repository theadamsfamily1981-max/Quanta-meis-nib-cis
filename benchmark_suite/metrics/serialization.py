"""Serialisation helpers for benchmark reporting."""

from __future__ import annotations

from typing import Dict, Mapping

from .base import BenchmarkReport
from .registry import MetricRegistry


def report_to_dict(report: BenchmarkReport) -> Dict[str, object]:
    """Public wrapper around :meth:`BenchmarkReport.to_dict`."""

    return report.to_dict()


def registry_to_payload(registry: MetricRegistry) -> Dict[str, object]:
    """Serialise a registry to a dictionary payload suitable for JSON output."""

    return registry.to_dict()


def merge_payload_with_metadata(
    payload: Dict[str, object],
    *,
    metadata: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    """Return a new payload dictionary with metadata merged in."""

    merged = dict(payload)
    if metadata:
        merged.setdefault("metadata", {}).update(dict(metadata))
    return merged
