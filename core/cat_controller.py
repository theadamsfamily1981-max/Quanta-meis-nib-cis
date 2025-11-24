"""Category controller used by the GRTES Phase III prototype.

The controller consumes Φ-extrapolation diagnostics and derives actionable
category assignments.  The prototype mirrors the high level behaviour of the
production controller which continuously toggles the neuromorphic hardware
between low-, medium- and high-energy modes depending on the projected Φ.

The implementation emphasises clarity so the experimentation team can plug the
module into notebooks without external dependencies.  The public API is tiny:
`CategoryController.evaluate` accepts a :class:`~core.phi_extrap.PhiExtrapolation`
and returns a :class:`CategoryDecision` describing the next mode to activate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List

from .phi_extrap import PhiExtrapolation


class CategoryLevel(str, Enum):
    """Discrete energy categories understood by the hardware adapter."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class CategoryBand:
    """Definition of a category band.

    Attributes
    ----------
    level:
        The :class:`CategoryLevel` assigned to the band.
    min_phi:
        Inclusive lower bound of Φ values that belong to the band.
    max_phi:
        Exclusive upper bound of Φ values that belong to the band.
    """

    level: CategoryLevel
    min_phi: float
    max_phi: float

    def contains(self, phi: float) -> bool:
        return self.min_phi <= phi < self.max_phi


@dataclass(frozen=True)
class CategoryDecision:
    """Result produced by the controller."""

    level: CategoryLevel
    rationale: str
    confidence: float


class CategoryController:
    """Map Φ extrapolations to hardware categories.

    The controller keeps a configurable list of :class:`CategoryBand` entries.
    Bands are evaluated in order; the first match wins.  A fallback ``LOW``
    band ensures that the evaluation always succeeds.
    """

    def __init__(self, bands: Iterable[CategoryBand] | None = None) -> None:
        if bands is None:
            bands = [
                CategoryBand(CategoryLevel.LOW, float("-inf"), 0.25),
                CategoryBand(CategoryLevel.MEDIUM, 0.25, 0.7),
                CategoryBand(CategoryLevel.HIGH, 0.7, float("inf")),
            ]
        self._bands: List[CategoryBand] = list(bands)
        if not self._bands:
            raise ValueError("CategoryController requires at least one band")

    @property
    def bands(self) -> List[CategoryBand]:
        return list(self._bands)

    def evaluate(self, extrapolation: PhiExtrapolation) -> CategoryDecision:
        """Determine the category for ``extrapolation``."""

        for band in self._bands:
            if band.contains(extrapolation.phi):
                rationale = self._build_rationale(band, extrapolation)
                return CategoryDecision(
                    level=band.level,
                    rationale=rationale,
                    confidence=extrapolation.confidence,
                )
        # Control should never reach this line because we always configure a
        # fallback band, but we keep the failure explicit for debugging.
        raise RuntimeError("No matching category band for extrapolation")

    @staticmethod
    def _build_rationale(band: CategoryBand, extrapolation: PhiExtrapolation) -> str:
        trend = "rising" if extrapolation.slope > 0 else "falling" if extrapolation.slope < 0 else "stable"
        return (
            f"Φ={extrapolation.phi:.3f} within [{band.min_phi}, {band.max_phi}) → "
            f"{band.level} (trend {trend}, confidence {extrapolation.confidence:.2f})"
        )


__all__ = [
    "CategoryController",
    "CategoryDecision",
    "CategoryBand",
    "CategoryLevel",
]
