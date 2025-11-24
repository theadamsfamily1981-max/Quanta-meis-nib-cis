"""Minimal JSONL telemetry logger."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
import json
import time


@dataclass
class JsonlLogger:
    """Append-only JSONL logger with a single schema."""

    path: Path
    schema: Dict[str, Any]
    auto_timestamp: bool = True
    _file: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf8")

    def log(self, **payload: Any) -> None:
        entry = dict(self.schema)
        entry.update(payload)
        if self.auto_timestamp:
            entry.setdefault("timestamp", time.time())
        self._file.write(json.dumps(entry, sort_keys=True) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["JsonlLogger"]
