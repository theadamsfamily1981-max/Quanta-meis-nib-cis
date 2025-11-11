"""Component implementations for TFAN v2."""

from .topology import TopologicalSignature, build_persistence_image, summarise_points
from .ssa import SSAConfig, SelectiveSelfAttention
from .fdr import (
    HutchinsonCurvatureEstimator,
    WeightFluctuationTracker,
    FDRControlSignal,
    control_from_metrics,
)
from .fep import FEPLossBreakdown, variational_free_energy

__all__ = [
    "TopologicalSignature",
    "build_persistence_image",
    "summarise_points",
    "SSAConfig",
    "SelectiveSelfAttention",
    "HutchinsonCurvatureEstimator",
    "WeightFluctuationTracker",
    "FDRControlSignal",
    "control_from_metrics",
    "FEPLossBreakdown",
    "variational_free_energy",
]
