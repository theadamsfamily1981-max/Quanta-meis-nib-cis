"""Logging utility scaffolding."""

from __future__ import annotations

from typing import Any, Dict


class EventLogger:
    """Minimal event logger for experimentation."""

    def __init__(self) -> None:
        self.records = []

    def log(self, level: str, message: str, **metadata: Any) -> None:
        """Log a message with optional metadata."""

        entry: Dict[str, Any] = {"level": level, "message": message, "metadata": metadata}
        self.records.append(entry)

    def dump(self) -> Dict[str, Any]:
        """Return all stored log records."""

        return {"records": list(self.records)}
