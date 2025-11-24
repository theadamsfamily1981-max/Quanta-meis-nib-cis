"""Tools for computing Φ-extrapolations for the GRTES prototype.

The module implements a tiny linear regression based extrapolator.  The goal
is not raw performance but instead a deterministic, well documented reference
implementation that other parts of the prototype can build upon.  The
extrapolator converts a numeric history into projected future Φ (phi) values
which we use to steer the category controller and hardware simulations.

A Φ-extrapolation is modelled as a linear trend prediction.  Even though the
real system ultimately plugs in a more sophisticated Bayesian estimator the
linear model gives us an intuitive and easy to verify baseline.  The
implementation includes light smoothing and diagnostic information so that
higher level orchestration code can understand the confidence of the
prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
import math


@dataclass(frozen=True)
class PhiSample:
    """A single observed Φ sample.

    Attributes
    ----------
    index:
        The sample position inside the time series.  We always index starting
        from zero so the first observation is at index ``0``.
    value:
        The Φ value that was measured.
    """

    index: int
    value: float


@dataclass(frozen=True)
class PhiExtrapolation:
    """Container for the extrapolated values and diagnostics.

    Attributes
    ----------
    phi:
        The projected Φ value ``future_steps`` points ahead of the last
        observation.
    slope:
        Estimated slope of the Φ trend per sample.
    intercept:
        Estimated intercept of the linear model.
    residual_error:
        Root mean squared error between the observed values and the fitted
        line.  A higher value indicates less confidence in the extrapolation.
    steps:
        Number of historical observations that were used for the fit.
    """

    phi: float
    slope: float
    intercept: float
    residual_error: float
    steps: int

    @property
    def confidence(self) -> float:
        """Return a heuristic confidence score in the range ``[0, 1]``.

        The confidence is derived from the residual error relative to the
        magnitude of the signal.  A perfectly linear signal will produce a
        confidence of ``1.0`` while very noisy signals approach ``0.0``.
        """

        if self.steps <= 1:
            return 0.0

        magnitude = abs(self.phi) + abs(self.intercept)
        if magnitude == 0:
            magnitude = 1.0
        normalized_error = min(self.residual_error / magnitude, 1.0)
        return max(0.0, 1.0 - normalized_error)


class PhiExtrapolator:
    """Perform Φ extrapolations on streaming numeric data."""

    def __init__(self, smoothing_window: int = 3) -> None:
        if smoothing_window < 1:
            raise ValueError("smoothing_window must be >= 1")
        self._smoothing_window = smoothing_window
        self._history: List[PhiSample] = []

    @staticmethod
    def _linear_regression(points: Sequence[PhiSample]) -> Tuple[float, float]:
        """Return ``(slope, intercept)`` for the given points."""

        n = len(points)
        if n == 0:
            raise ValueError("Cannot regress an empty set of points")
        sum_x = sum(p.index for p in points)
        sum_y = sum(p.value for p in points)
        sum_xx = sum(p.index * p.index for p in points)
        sum_xy = sum(p.index * p.value for p in points)

        denominator = n * sum_xx - sum_x * sum_x
        if math.isclose(denominator, 0.0):
            # All indices might be identical; fall back to zero slope.
            slope = 0.0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        return slope, intercept

    @staticmethod
    def _rmse(points: Sequence[PhiSample], slope: float, intercept: float) -> float:
        """Compute the root mean squared error for the fitted line."""

        if not points:
            return 0.0
        error = sum((p.value - (slope * p.index + intercept)) ** 2 for p in points)
        return math.sqrt(error / len(points))

    def _smoothed_history(self) -> List[PhiSample]:
        """Return a smoothed version of the historical Φ values.

        The smoothing uses a simple moving average over ``smoothing_window``
        samples to stabilize noisy measurements.  The method keeps the original
        indices intact so downstream regression remains meaningful.
        """

        if len(self._history) <= self._smoothing_window:
            return list(self._history)

        window = self._smoothing_window
        smoothed: List[PhiSample] = []
        values = [sample.value for sample in self._history]
        for idx, sample in enumerate(self._history):
            start = max(0, idx - window + 1)
            window_values = values[start : idx + 1]
            smoothed.append(PhiSample(index=sample.index, value=sum(window_values) / len(window_values)))
        return smoothed

    def observe(self, value: float) -> None:
        """Record a new Φ observation."""

        index = len(self._history)
        self._history.append(PhiSample(index=index, value=float(value)))

    def extrapolate(self, future_steps: int = 5) -> PhiExtrapolation:
        """Compute the Φ extrapolation ``future_steps`` ahead of the last sample."""

        if not self._history:
            raise ValueError("Cannot extrapolate without observations")

        smoothed = self._smoothed_history()
        slope, intercept = self._linear_regression(smoothed)
        last_index = smoothed[-1].index
        target_index = last_index + future_steps
        phi_value = slope * target_index + intercept
        residual = self._rmse(smoothed, slope, intercept)
        return PhiExtrapolation(
            phi=phi_value,
            slope=slope,
            intercept=intercept,
            residual_error=residual,
            steps=len(smoothed),
        )

    def clear(self) -> None:
        """Reset the extrapolator history."""

        self._history.clear()

    def __len__(self) -> int:  # pragma: no cover - trivial delegation
        return len(self._history)


def extrapolate_phi(values: Iterable[float], future_steps: int = 5) -> PhiExtrapolation:
    """Convenience function to compute a Φ extrapolation from ``values``."""

    extrapolator = PhiExtrapolator()
    for value in values:
        extrapolator.observe(value)
    return extrapolator.extrapolate(future_steps=future_steps)
