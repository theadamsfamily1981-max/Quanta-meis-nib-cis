"""MEIS Core Module.

This module implements the core MEIS (Meta-Epistemic Inference System)
logic used by the QUANTA–TFAN unified framework. The implementation is
kept intentionally lightweight so it can be executed in both research and
production environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence


@dataclass
class Hypothesis:
    """Represents a candidate hypothesis considered by MEIS."""

    name: str
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_confidence(self, observation_score: float) -> None:
        """Update confidence using a multiplicative update rule."""

        prior = self.confidence
        self.confidence = max(min(prior * observation_score, 1.0), 0.0)
        self.metadata.setdefault("history", []).append(
            {
                "prior": prior,
                "observation": observation_score,
                "posterior": self.confidence,
            }
        )


class MEISEngine:
    """A tiny but functional implementation of the MEIS inference loop."""

    def __init__(self, hypotheses: Sequence[Hypothesis]):
        if not hypotheses:
            raise ValueError("MEIS requires at least one hypothesis")
        self._hypotheses: List[Hypothesis] = list(hypotheses)

    @property
    def hypotheses(self) -> Sequence[Hypothesis]:
        return tuple(self._hypotheses)

    def infer(self, evidence_stream: Sequence[float]) -> Hypothesis:
        """Run an inference pass over the supplied evidence scores."""

        for score in evidence_stream:
            for hypothesis in self._hypotheses:
                hypothesis.update_confidence(score)
        return max(self._hypotheses, key=lambda h: h.confidence)


def run_meis_inference(scores: Sequence[float]) -> Hypothesis:
    """Convenience wrapper for quick experiments."""

    default_hypotheses = [
        Hypothesis("stability"),
        Hypothesis("plasticity", 0.75),
        Hypothesis("antifragility", 0.6),
    ]
    engine = MEISEngine(default_hypotheses)
    return engine.infer(scores)
