"""TFAN Phase I utilities.

This module provides the initial scaffolding for TFAN phase work.
"""

from __future__ import annotations


def summarize_inputs(inputs: list[float]) -> float:
    """Return the average of the provided numeric inputs.

    Parameters
    ----------
    inputs:
        A list of numeric values representing the initial TFAN probes.

    Returns
    -------
    float
        The arithmetic mean of the provided inputs. If ``inputs`` is empty,
        ``0.0`` is returned to signal an idle phase.
    """

    if not inputs:
        return 0.0
    return sum(inputs) / len(inputs)
