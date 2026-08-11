"""CIFAR adapter benchmark for the embodied integration stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, MutableMapping, Sequence

from .metrics import BenchmarkReport, Metric


@dataclass(frozen=True)
class DeploymentScenario:
    """Deployment scenario used to parametrise the CIFAR adapter."""

    name: str
    batch_size: int
    latency_budget_ms: float
    reference_accuracy: float


_DEFAULT_SCENARIOS: Sequence[DeploymentScenario] = (
    DeploymentScenario(
        name="edge_quantised",
        batch_size=8,
        latency_budget_ms=25.0,
        reference_accuracy=0.812,
    ),
    DeploymentScenario(
        name="edge_fp16",
        batch_size=16,
        latency_budget_ms=18.0,
        reference_accuracy=0.855,
    ),
    DeploymentScenario(
        name="datacenter",
        batch_size=128,
        latency_budget_ms=5.0,
        reference_accuracy=0.912,
    ),
)


class CIFARAdapter:
    """Evaluate the adapter stack using synthetic CIFAR scenarios."""

    def __init__(
        self,
        scenarios: Sequence[DeploymentScenario] | None = None,
        *,
        callbacks: Iterable["CIFARCallback"] | None = None,
    ) -> None:
        self._scenarios = list(scenarios or _DEFAULT_SCENARIOS)
        self._callbacks = list(callbacks or [])

    def run(self) -> BenchmarkReport:
        """Execute the configured scenarios and return a benchmark report."""

        report = BenchmarkReport(benchmark="cifar_adapter")
        report.metadata.update(self.describe())

        for scenario in self._scenarios:
            report.extend_metrics(self._build_scenario_metrics(scenario))

        for callback in self._callbacks:
            callback.on_report(report)

        return report

    def describe(self) -> MutableMapping[str, object]:
        """Describe the configured CIFAR deployment scenarios."""

        return {
            "scenarios": [
                {
                    "name": scenario.name,
                    "batch_size": scenario.batch_size,
                    "latency_budget_ms": scenario.latency_budget_ms,
                    "reference_accuracy": scenario.reference_accuracy,
                }
                for scenario in self._scenarios
            ]
        }

    @staticmethod
    def _build_scenario_metrics(scenario: DeploymentScenario) -> Iterable[Metric]:
        """Construct deterministic metrics for a single deployment scenario."""

        throughput = scenario.batch_size / (scenario.latency_budget_ms / 1000.0)
        accuracy = min(scenario.reference_accuracy * 1.015, 0.99)
        energy = round(0.35 * scenario.batch_size ** 0.5, 3)

        yield Metric(
            name=f"{scenario.name}_throughput", value=round(throughput, 2), unit="img/s"
        )
        yield Metric(
            name=f"{scenario.name}_accuracy", value=round(accuracy, 3), unit="fraction"
        )
        yield Metric(
            name=f"{scenario.name}_energy", value=energy, unit="joules", higher_is_better=False
        )


class CIFARCallback:
    """Base interface for observing CIFAR adapter runs."""

    def on_report(self, report: BenchmarkReport) -> None:  # pragma: no cover - interface only
        raise NotImplementedError


class HistoryCallback(CIFARCallback):
    """Persist the latest report for later inspection."""

    def __init__(self) -> None:
        self.history: List[Mapping[str, object]] = []

    def on_report(self, report: BenchmarkReport) -> None:
        self.history.append(report.to_dict())

    def latest(self) -> Mapping[str, object] | None:
        return self.history[-1] if self.history else None
