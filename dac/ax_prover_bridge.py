"""Bridge between DAC structures and a hypothetical prover."""
from __future__ import annotations

from typing import Iterable

from .kripke_state import KripkeState


def certify_axioms(state: KripkeState, required: Iterable[str]) -> bool:
    """Return ``True`` when all required propositions hold in the state."""

    return state.evaluate({symbol: True for symbol in required})


__all__ = ["certify_axioms"]
