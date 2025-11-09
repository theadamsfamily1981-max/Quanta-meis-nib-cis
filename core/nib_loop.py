"""NIB (Neural Integration Bridge) loop implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List


@dataclass
class NIBConfig:
    """Configuration for the NIB loop."""

    window_size: int = 4
    smoothing_factor: float = 0.6


class NIBLoop:
    """Minimal neural integration bridge loop."""

    def __init__(self, config: NIBConfig | None = None):
        self.config = config or NIBConfig()
        self._buffer: List[float] = []

    def push(self, value: float) -> float:
        """Push a new value through the bridge and return the smoothed output."""

        self._buffer.append(value)
        if len(self._buffer) > self.config.window_size:
            self._buffer.pop(0)
        average = sum(self._buffer) / len(self._buffer)
        smoothed = (
            self.config.smoothing_factor * average
            + (1 - self.config.smoothing_factor) * value
        )
        return smoothed

    def run(self, stream: Iterable[float], callback: Callable[[float], None]) -> None:
        """Run the loop over a stream of values and emit via callback."""

        for value in stream:
            callback(self.push(value))
