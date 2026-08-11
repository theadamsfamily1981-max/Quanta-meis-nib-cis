"""Topological Feature Attention Network (TFAN) v2 package."""

from .components import (
    TopologicalSignature,
    build_persistence_image,
    summarise_points,
    SSAConfig,
    SelectiveSelfAttention,
    HutchinsonCurvatureEstimator,
    WeightFluctuationTracker,
    FDRControlSignal,
    control_from_metrics,
    FEPLossBreakdown,
    variational_free_energy,
)
from .policies import TLSPolicy, FDRLRScheduler, EarlyStopController

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
    "TLSPolicy",
    "FDRLRScheduler",
    "EarlyStopController",
]
