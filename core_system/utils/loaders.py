"""Data loader scaffolding."""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Sequence


def from_sequence(sequence: Sequence[Any]) -> Iterator[Any]:
    """Yield items from a static sequence as a simple data loader."""

    for item in sequence:
        yield item
