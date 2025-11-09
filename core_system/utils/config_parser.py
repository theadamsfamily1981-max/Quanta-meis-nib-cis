"""Configuration parsing scaffolding."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
