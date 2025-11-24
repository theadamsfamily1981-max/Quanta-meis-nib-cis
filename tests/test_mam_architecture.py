"""Smoke tests for the MAM architecture."""
from __future__ import annotations

from typing import Any, List

import pytest

from meis.mam import MAMArchitecture, MAMRouter


def doubling_module(inputs: Any) -> Any:
    return inputs * 2


def stringify_module(inputs: Any) -> str:
    return f"value:{inputs}"


def test_architecture_runs_all_modules() -> None:
    router = MAMRouter()
    architecture = MAMArchitecture(modules=[doubling_module, stringify_module], router=router)

    result = architecture.run(3)

    assert result == [6, "value:3"]


def test_architecture_requires_modules() -> None:
    router = MAMRouter()

    with pytest.raises(ValueError):
        MAMArchitecture(modules=[], router=router)


def test_architecture_is_callable() -> None:
    router = MAMRouter()
    architecture = MAMArchitecture(modules=[doubling_module], router=router)

    assert architecture(4) == [8]
