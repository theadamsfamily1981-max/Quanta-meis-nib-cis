"""Routing strategies for the MAM architecture."""
from __future__ import annotations

from typing import Any, Iterable, List, Protocol, Sequence


class RoutingStrategy(Protocol):
    """Protocol describing how modules should be selected for an input."""

    def select(self, inputs: Any, num_modules: int) -> Iterable[int]:
        """Return an iterable of indices indicating which modules to execute."""


class SequentialRouting:
    """Simple routing strategy that sequentially executes every module once."""

    def select(self, inputs: Any, num_modules: int) -> Iterable[int]:
        return range(num_modules)


class MAMRouter:
    """Router that coordinates module execution using a routing strategy."""

    def __init__(self, strategy: RoutingStrategy | None = None) -> None:
        self._strategy = strategy or SequentialRouting()

    def route(self, inputs: Any, modules: Sequence[Any]) -> List[Any]:
        outputs: List[Any] = []
        for index in self._strategy.select(inputs, len(modules)):
            outputs.append(modules[index](inputs))
        return outputs
