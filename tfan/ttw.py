"""Trigger wrapper utilities."""
from __future__ import annotations

from typing import Callable, Iterable, Sequence


class TriggeredWrapper:
    """Light-weight orchestration helper for triggerable callbacks."""

    def __init__(self, callbacks: Iterable[Callable[[Sequence[float]], None]] | None = None) -> None:
        self._callbacks = list(callbacks or [])

    def register(self, callback: Callable[[Sequence[float]], None]) -> None:
        self._callbacks.append(callback)

    def fire(self, values: Sequence[float]) -> None:
        snapshot = list(values)
        for callback in self._callbacks:
            callback(snapshot)

    def clear(self) -> None:
        self._callbacks.clear()


__all__ = ["TriggeredWrapper"]
