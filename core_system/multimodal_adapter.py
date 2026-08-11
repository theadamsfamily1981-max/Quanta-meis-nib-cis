"""Multimodal adapter hub scaffolding."""

from __future__ import annotations

from typing import Any, Dict


class MultimodalAdapter:
    """Manage adapters for text, image, and audio modalities."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adapters: Dict[str, Any] = {}

    def register(self, modality: str, adapter: Any) -> None:
        """Register a modality-specific adapter."""

        self.adapters[modality] = adapter

    def transform(self, modality: str, data: Any) -> Any:
        """Apply the adapter for a given modality."""

        adapter = self.adapters.get(modality)
        if adapter is None:
            raise KeyError(f"No adapter registered for modality '{modality}'")
        return adapter(data)

    def available_modalities(self) -> Dict[str, Any]:
        """Return a mapping of modalities to their adapters."""

        return dict(self.adapters)
