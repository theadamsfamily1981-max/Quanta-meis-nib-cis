"""Module orchestration and event routing scaffolding."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class MEISOrchestrator:
    """Coordinate communication between the core subsystems."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.event_log: List[Dict[str, Any]] = []
        self.modules: Dict[str, Any] = {}

    def register_module(self, name: str, module: Any) -> None:
        """Register a module for orchestration."""

        self.modules[name] = module

    def emit(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Emit an event to the registered modules."""

        record = {"event": event, "payload": payload or {}}
        self.event_log.append(record)

    def route(self, event: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Route events to target modules and collect responses."""

        self.emit(event, payload)
        responses: Dict[str, Any] = {}
        for name, module in self.modules.items():
            handler = getattr(module, "handle_event", None)
            if callable(handler):
                responses[name] = handler(event, payload)
        return responses

    def history(self) -> List[Dict[str, Any]]:
        """Return a copy of the orchestration event history."""

        return list(self.event_log)
