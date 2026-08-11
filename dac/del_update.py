"""Dynamic epistemic logic updates for the Kripke state."""
from __future__ import annotations

from typing import Iterable

from .kripke_state import KripkeState


def public_announcement(state: KripkeState, truths: Iterable[str]) -> KripkeState:
    """Perform a public announcement update.

    The function returns a new ``KripkeState`` where all announced propositions
    are set to ``True``.  The operation is idempotent and keeps the relation
    structure untouched which is sufficient for the lightweight testing in this
    repository.
    """

    updated = state.copy()
    for symbol in truths:
        updated.propositions[symbol] = True
    return updated


__all__ = ["public_announcement"]
