"""Utility package for topology-first attention networks.

The :mod:`tfan` namespace exposes a collection of light-weight helpers that are
used across experiments in this repository.  Only a tiny subset of a research
codebase is provided here – the focus is on deterministic, well documented
utilities that the tests rely on.

All public symbols are re-exported from the submodules so users may simply::

    from tfan import TopologicalLandmarkSelector, selective_self_attention

without having to know the individual module layout.
"""

from .tls import TopologicalLandmarkSelector, LandmarkSelectionResult
from .ssa import selective_self_attention, temperature_scaled_softmax
from .fdr import (
    benjamini_hochberg,
    benjamini_yekutieli,
    estimate_fdr_thresholds,
    suggest_lr_and_temperature,
)
from .topo import persistent_surprise, proxy_persistence_diagram
from .ctd import CurvatureGate
from .pgu import ProofGoalCache, Z3LiteProofChecker
from .ttw import TriggeredWrapper
from .multiobj import pareto_front, expected_hypervolume_improvement
from .telemetry import JsonlLogger

__all__ = [
    "TopologicalLandmarkSelector",
    "LandmarkSelectionResult",
    "selective_self_attention",
    "temperature_scaled_softmax",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "estimate_fdr_thresholds",
    "suggest_lr_and_temperature",
    "persistent_surprise",
    "proxy_persistence_diagram",
    "CurvatureGate",
    "ProofGoalCache",
    "Z3LiteProofChecker",
    "TriggeredWrapper",
    "pareto_front",
    "expected_hypervolume_improvement",
    "JsonlLogger",
]
