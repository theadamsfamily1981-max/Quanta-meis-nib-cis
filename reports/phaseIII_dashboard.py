"""Phase III visualization dashboard utilities.

This module exposes a small façade around Matplotlib/Pandas so the
research team can quickly inspect curvature, persistence and
:math:`\Phi_{extrap}` trends captured during Phase III experiments.

The helper is intentionally lightweight; it accepts a pre-loaded
``pandas.DataFrame`` instance to avoid imposing any I/O strategy on the
caller, while still providing a convenience classmethod for CSV based
workflows.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Optional

import matplotlib.pyplot as plt
import pandas as pd

# Column names used throughout the dashboard. Keeping them in a constant makes
# it easy to reference and adjust in one place should the acquisition pipeline
# change.
TIMESTAMP_COL = "timestamp"
CURVATURE_COL = "curvature"
PERSISTENCE_COL = "persistence"
PHI_EXTRAP_COL = "phi_extrap"


class DashboardDataError(ValueError):
    """Raised when the provided dataset is missing required information."""


@dataclass
class PhaseIIIDashboard:
    """Small helper to render Phase III monitoring visualisations.

    Parameters
    ----------
    frame:
        A tidy :class:`~pandas.DataFrame` containing at least the columns
        required for the plots emitted by this class:

        * ``timestamp`` – the acquisition time (string or datetime)
        * ``curvature`` – the curvature measurement (float)
        * ``persistence`` – the persistence metric (float)
        * ``phi_extrap`` – extrapolated potential :math:`\Phi` (float)
    """

    frame: pd.DataFrame

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_csv(
        cls,
        path: Path | str,
        *,
        parse_dates: Optional[Iterable[str]] = (TIMESTAMP_COL,),
        **kwargs,
    ) -> "PhaseIIIDashboard":
        """Build a dashboard from a CSV file.

        Parameters
        ----------
        path:
            Location of the CSV file to load.
        parse_dates:
            Optional iterable describing which columns should be parsed as
            datetimes. By default we parse the ``timestamp`` column to make it
            easier to generate timeline plots.
        kwargs:
            Additional keyword arguments forwarded to
            :func:`pandas.read_csv`.
        """

        frame = pd.read_csv(path, parse_dates=parse_dates, **kwargs)
        return cls(frame)

    # ------------------------------------------------------------------
    # Validation utilities
    # ------------------------------------------------------------------
    def _require_columns(self, *columns: str) -> None:
        """Ensure that all columns required for a plot are present."""

        missing = [column for column in columns if column not in self.frame]
        if missing:
            raise DashboardDataError(
                "Dataset is missing required columns: " + ", ".join(missing)
            )

    # ------------------------------------------------------------------
    # Plot builders
    # ------------------------------------------------------------------
    def plot_curvature_vs_persistence(self, *, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot curvature against persistence as a scatter chart.

        Highlights the correlation between our geometric and topological
        descriptors for the experiment run.
        """

        self._require_columns(TIMESTAMP_COL, CURVATURE_COL, PERSISTENCE_COL)
        ax = ax or plt.gca()
        ax.scatter(
            self.frame[CURVATURE_COL],
            self.frame[PERSISTENCE_COL],
            c=self._timestamp_as_numeric(),
            cmap="viridis",
            alpha=0.75,
            edgecolor="k",
        )
        ax.set_xlabel("Curvature")
        ax.set_ylabel("Persistence")
        ax.set_title("Curvature vs Persistence – Phase III")
        sm = plt.cm.ScalarMappable(cmap="viridis")
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Timeline index")
        return ax

    def plot_phi_extrap_timeline(self, *, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot the :math:`\Phi_{extrap}` timeline.

        Displays the extrapolated potential values against time, providing
        a quick glance at trend changes or outliers.
        """

        self._require_columns(TIMESTAMP_COL, PHI_EXTRAP_COL)
        ax = ax or plt.gca()
        sorted_frame = self.frame.sort_values(TIMESTAMP_COL)
        ax.plot(sorted_frame[TIMESTAMP_COL], sorted_frame[PHI_EXTRAP_COL], marker="o")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Φ_extrap")
        ax.set_title("Φ_extrap timeline – Phase III")
        ax.grid(True, linestyle="--", alpha=0.4)
        return ax

    def _timestamp_as_numeric(self) -> pd.Series:
        """Convert the timestamp column into an integer index.

        The scatter plot uses this helper to map time progression to colour.
        """

        values = self.frame[TIMESTAMP_COL]
        if not pd.api.types.is_datetime64_any_dtype(values):
            # When the timestamps are not parsed as datetimes we simply fall back
            # to the index ordering. This keeps the scatter plot usable without
            # forcing strict typing on the caller.
            return pd.Series(range(len(values)), index=values.index)
        return (values - values.min()).dt.total_seconds()

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------
    def summary(self) -> Mapping[str, float]:
        """Compute a compact statistical summary for the dashboard."""

        self._require_columns(CURVATURE_COL, PERSISTENCE_COL, PHI_EXTRAP_COL)
        stats: MutableMapping[str, float] = {}
        stats["curvature_mean"] = float(self.frame[CURVATURE_COL].mean())
        stats["curvature_std"] = float(self.frame[CURVATURE_COL].std())
        stats["persistence_mean"] = float(self.frame[PERSISTENCE_COL].mean())
        stats["persistence_std"] = float(self.frame[PERSISTENCE_COL].std())
        stats["phi_extrap_mean"] = float(self.frame[PHI_EXTRAP_COL].mean())
        stats["phi_extrap_std"] = float(self.frame[PHI_EXTRAP_COL].std())
        return stats

    def render_dashboard(self, output: Path | str) -> None:
        """Render both plots to a single figure and persist it to ``output``."""

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax_scatter, ax_timeline) = plt.subplots(1, 2, figsize=(16, 6))
        try:
            self.plot_curvature_vs_persistence(ax=ax_scatter)
            self.plot_phi_extrap_timeline(ax=ax_timeline)
            fig.tight_layout()
            fig.savefig(output_path)
        finally:
            plt.close(fig)


__all__ = [
    "PhaseIIIDashboard",
    "DashboardDataError",
    "TIMESTAMP_COL",
    "CURVATURE_COL",
    "PERSISTENCE_COL",
    "PHI_EXTRAP_COL",
]
