"""Base data structures for benchmark metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional


@dataclass(frozen=True)
class Metric:
    """Represents a single evaluation metric.

    Attributes
    ----------
    name:
        Canonical metric name.
    value:
        Numerical score associated with the metric.
    higher_is_better:
        Flag indicating whether larger values are preferred.
    unit:
        Optional unit associated with the metric value.
    metadata:
        Optional dictionary containing additional context for the metric.
    """

    name: str
    value: float
    higher_is_better: bool = True
    unit: Optional[str] = None
    metadata: Mapping[str, object] | None = None

    def to_dict(self) -> Dict[str, object]:
        """Convert the metric to a JSON-serialisable dictionary."""

        payload: Dict[str, object] = {
            "name": self.name,
            "value": self.value,
            "higher_is_better": self.higher_is_better,
        }
        if self.unit is not None:
            payload["unit"] = self.unit
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass
class BenchmarkReport:
    """Collection of metrics generated during a benchmark run."""

    benchmark: str
    metrics: List[Metric] = field(default_factory=list)
    metadata: MutableMapping[str, object] = field(default_factory=dict)

    def add_metric(self, metric: Metric) -> None:
        """Append a metric to the report."""

        self.metrics.append(metric)

    def extend_metrics(self, metrics: Iterable[Metric]) -> None:
        """Append multiple metrics to the report."""

        self.metrics.extend(metrics)

    def to_dict(self) -> Dict[str, object]:
        """Serialise the report into a dictionary."""

        return {
            "benchmark": self.benchmark,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "metadata": dict(self.metadata),
        }
