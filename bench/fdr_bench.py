"""Utility for evaluating FDR detectors."""
from __future__ import annotations

from typing import Sequence

from tfan.fdr import estimate_fdr_thresholds


def auroc_for_stationarity(p_values: Sequence[float], *, alpha: float = 0.05) -> float:
    """Return a toy AUROC style metric for stationarity detection."""

    result = estimate_fdr_thresholds(p_values, alpha=alpha)
    discoveries = len(result.discoveries)
    if discoveries == 0:
        return 0.5
    return min(1.0, 0.5 + discoveries / len(p_values))


__all__ = ["auroc_for_stationarity"]


if __name__ == "__main__":
    toy = [0.01, 0.02, 0.2, 0.3, 0.7]
    print(f"stationarity AUROC≈{auroc_for_stationarity(toy):.3f}")
