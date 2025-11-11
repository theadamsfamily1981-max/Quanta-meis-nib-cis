"""Kripke style state containers for the dynamic action calculus."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, MutableMapping, Set


@dataclass
class KripkeState:
    """Minimal representation of modal worlds used by the DAC bridge."""

    propositions: MutableMapping[str, bool] = field(default_factory=dict)
    relations: MutableMapping[str, Set[str]] = field(default_factory=dict)

    def copy(self) -> "KripkeState":
        return KripkeState(
            propositions=dict(self.propositions),
            relations={k: set(v) for k, v in self.relations.items()},
        )

    def add_relation(self, source: str, targets: Iterable[str]) -> None:
        rel = self.relations.setdefault(source, set())
        rel.update(targets)

    def evaluate(self, assumptions: Mapping[str, bool]) -> bool:
        return all(self.propositions.get(name, False) == value for name, value in assumptions.items())


__all__ = ["KripkeState"]
