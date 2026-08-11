"""Unified refinement driver for Phase 2 validation.

This script consolidates the persistent homology (PH) pipeline checks
introduced in Phase 2. It is intentionally lightweight so that the
critical logic is transparent to reviewers of the milestone branch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "open_problems_phase2_results.json"


@dataclass(frozen=True)
class ValidationMetric:
    """Structured representation of a single validation metric."""

    name: str
    achieved: float
    target: float
    unit: str
    higher_is_better: bool = False

    def passes(self) -> bool:
        """Return True when the achieved value satisfies the target bound."""
        if self.higher_is_better:
            return self.achieved >= self.target
        return self.achieved <= self.target

    def summary(self) -> str:
        status = "PASS" if self.passes() else "FAIL"
        return f"[{status}] {self.name}: {self.achieved}{self.unit} (target {self.target}{self.unit})"


@dataclass(frozen=True)
class PhaseTwoReport:
    pass_rate: float
    benchmarks_passed: int
    benchmarks_total: int
    metrics: List[ValidationMetric]
    upgrades: Tuple[str, ...]

    @classmethod
    def from_json(cls, payload: dict) -> "PhaseTwoReport":
        summary = payload["summary"]
        performance = payload["performance"]
        metrics = [
            ValidationMetric("latency", max(performance["real_time_ph_ms"]), 10.0, " ms"),
            ValidationMetric("noise_consistency", performance["noise_robustness_consistency"], 0.99, "", True),
            ValidationMetric("scale_invariance_drift", performance["scale_invariance_drift"], 0.005, ""),
            ValidationMetric("multi_scale_stability", performance["multi_scale_stability"], 0.98, "", True),
            ValidationMetric("metastability_cv", performance["metastability_cv"], 0.30, ""),
            ValidationMetric("pass_rate", summary["pass_rate"], 0.667, "", True),
        ]
        upgrades = tuple(payload.get("upgrades", []))
        return cls(
            pass_rate=summary["pass_rate"],
            benchmarks_passed=summary["benchmarks_passed"],
            benchmarks_total=summary["benchmarks_total"],
            metrics=metrics,
            upgrades=upgrades,
        )

    def render(self) -> str:
        header = (
            f"Phase 2 validation: {self.benchmarks_passed}/{self.benchmarks_total} "
            f"benchmarks passed ({self.pass_rate:.1%})."
        )
        body = "\n".join(metric.summary() for metric in self.metrics)
        upgrades = "\n".join(f"- {upgrade}" for upgrade in self.upgrades)
        return f"{header}\n\nMetrics:\n{body}\n\nUpgrades:\n{upgrades}"


def load_results(path: Path = RESULTS_PATH) -> PhaseTwoReport:
    """Load the milestone results JSON and convert to a report."""
    payload = json.loads(path.read_text())
    return PhaseTwoReport.from_json(payload)


def main() -> None:
    report = load_results()
    print(report.render())


if __name__ == "__main__":  # pragma: no cover - manual driver
    main()
