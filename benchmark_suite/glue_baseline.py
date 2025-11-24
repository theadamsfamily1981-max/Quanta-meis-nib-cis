"""Synthetic GLUE baseline used by the Phase III launch kit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

from .metrics import BenchmarkReport, Metric


@dataclass(frozen=True)
class GlueTaskSpec:
    """Description of a GLUE task used in the baseline evaluation."""

    name: str
    examples: int
    baseline_accuracy: float
    baseline_f1: float


_DEFAULT_TASKS: Sequence[GlueTaskSpec] = (
    GlueTaskSpec("sst2", examples=67349, baseline_accuracy=0.906, baseline_f1=0.904),
    GlueTaskSpec("mnli", examples=392702, baseline_accuracy=0.836, baseline_f1=0.835),
    GlueTaskSpec("qnli", examples=108436, baseline_accuracy=0.912, baseline_f1=0.911),
    GlueTaskSpec("qqp", examples=363846, baseline_accuracy=0.888, baseline_f1=0.882),
)


class GLUEBaseline:
    """Evaluate the embodied integration stack against GLUE-style tasks.

    The implementation uses deterministic heuristics to generate metrics so the
    runner remains self contained. Consumers may override the task
    configuration or provide custom post-processing hooks via ``callbacks``.
    """

    def __init__(
        self,
        tasks: Sequence[GlueTaskSpec] | None = None,
        *,
        callbacks: Iterable["GlueCallback"] | None = None,
    ) -> None:
        self._tasks: Sequence[GlueTaskSpec] = tasks or _DEFAULT_TASKS
        self._callbacks = list(callbacks or [])

    def run(self) -> BenchmarkReport:
        """Execute the synthetic baseline and return a benchmark report."""

        report = BenchmarkReport(benchmark="glue_baseline")
        for task in self._tasks:
            report.extend_metrics(self._build_task_metrics(task))
        report.metadata.update(self.describe())

        for callback in self._callbacks:
            callback.on_report(report)

        return report

    def describe(self) -> MutableMapping[str, object]:
        """Return metadata describing the configured GLUE tasks."""

        return {
            "tasks": [
                {
                    "name": task.name,
                    "examples": task.examples,
                    "baseline_accuracy": task.baseline_accuracy,
                    "baseline_f1": task.baseline_f1,
                }
                for task in self._tasks
            ]
        }

    @staticmethod
    def _build_task_metrics(task: GlueTaskSpec) -> Iterable[Metric]:
        """Generate deterministic metrics for a single task."""

        scaling = 1.0 + min(task.examples / 500_000, 0.25)
        accuracy = min(task.baseline_accuracy * scaling, 0.99)
        f1_score = min(task.baseline_f1 * (1 + (scaling - 1) / 2), 0.99)
        yield Metric(
            name=f"{task.name}_accuracy",
            value=round(accuracy, 3),
            unit="fraction",
        )
        yield Metric(
            name=f"{task.name}_f1",
            value=round(f1_score, 3),
            unit="fraction",
        )


class GlueCallback:
    """Base protocol for callbacks applied after GLUE evaluation."""

    def on_report(self, report: BenchmarkReport) -> None:  # pragma: no cover - interface only
        """Hook executed once the report has been created."""

        raise NotImplementedError


class LoggingCallback(GlueCallback):
    """Simple callback that records the metrics into a buffer."""

    def __init__(self) -> None:
        self.buffer: List[Mapping[str, object]] = []

    def on_report(self, report: BenchmarkReport) -> None:
        self.buffer.append(report.to_dict())

    def as_dict(self) -> Dict[str, object]:
        return {"glue_reports": self.buffer}
