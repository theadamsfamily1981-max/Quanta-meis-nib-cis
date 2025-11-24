"""Foundational classes for the Mycelial-Adaptive Multimodal (MAM) stack."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Protocol, Sequence

from .routing import MAMRouter


class MAMModule(Protocol):
    """Protocol describing the call signature of a MAM processing unit."""

    def __call__(self, inputs: Any) -> Any:
        """Process ``inputs`` and return an output consumable by downstream modules."""


@dataclass
class MAMArchitecture:
    """Composable container coordinating a router with a set of modules."""

    modules: Sequence[MAMModule]
    router: MAMRouter
    aggregator: Callable[[Iterable[Any]], Any] = field(default=lambda outputs: list(outputs))

    def __post_init__(self) -> None:
        if not isinstance(self.modules, Sequence) or len(self.modules) == 0:
            raise ValueError("MAMArchitecture requires at least one module")

    def run(self, inputs: Any) -> Any:
        """Route ``inputs`` through the configured modules and aggregate outputs."""

        outputs: List[Any] = self.router.route(inputs, self.modules)
        return self.aggregator(outputs)

    def __call__(self, inputs: Any) -> Any:
        """Allow the architecture to be invoked like a function."""

        return self.run(inputs)
