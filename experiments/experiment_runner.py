"""Experiment runner orchestrating MEIS, NIB and T-FAN components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from core.meis import Hypothesis, MEISEngine
from core.nib_loop import NIBLoop, NIBConfig


class TFANModel(Protocol):
    def __call__(self, x, *args, **kwargs): ...


@dataclass
class ExperimentConfig:
    nib: NIBConfig
    evidence_stream: Iterable[float]


class ExperimentRunner:
    def __init__(
        self,
        meis_engine: MEISEngine,
        nib_loop: NIBLoop,
        tfan_model: TFANModel,
        config: ExperimentConfig,
    ):
        self.meis_engine = meis_engine
        self.nib_loop = nib_loop
        self.tfan_model = tfan_model
        self.config = config

    def run(self, data) -> Hypothesis:
        """Run a single experiment returning the highest-confidence hypothesis."""

        nib_outputs: list[float] = []
        self.nib_loop.run(self.config.evidence_stream, nib_outputs.append)
        _ = self.tfan_model(data)
        return self.meis_engine.infer(nib_outputs)

    @classmethod
    def create_default(cls, evidence_stream: Iterable[float], tfan_model: TFANModel) -> "ExperimentRunner":
        hypotheses = [Hypothesis("baseline"), Hypothesis("adaptation", 0.7)]
        meis = MEISEngine(hypotheses)
        nib_loop = NIBLoop(NIBConfig())
        config = ExperimentConfig(NIBConfig(), evidence_stream)
        return cls(meis, nib_loop, tfan_model, config)
