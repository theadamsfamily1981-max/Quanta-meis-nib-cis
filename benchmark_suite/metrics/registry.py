"""Metric registry and aggregation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Dict, Iterable, List, Mapping

from .base import BenchmarkReport, Metric


@dataclass
class MetricRegistry:
    """In-memory store for benchmark reports."""

    reports: List[BenchmarkReport] = field(default_factory=list)

    def register(self, report: BenchmarkReport) -> None:
        """Store a report in the registry."""

        self.reports.append(report)

    def extend(self, reports: Iterable[BenchmarkReport]) -> None:
        """Store multiple reports in the registry."""

        self.reports.extend(reports)

    def iter_metrics(self) -> Iterable[Metric]:
        """Iterate over all metrics contained within the registry."""

        for report in self.reports:
            yield from report.metrics

    def summary(self) -> Mapping[str, float]:
        """Generate a summary of the most common metrics."""

        grouped: Dict[str, List[float]] = {}
        for metric in self.iter_metrics():
            grouped.setdefault(metric.name, []).append(metric.value)

        return {name: fmean(values) for name, values in grouped.items() if values}

    def to_dict(self) -> Dict[str, object]:
        """Serialise the registry, including individual reports and summary."""

        return {
            "reports": [report.to_dict() for report in self.reports],
            "summary": dict(self.summary()),
        }
