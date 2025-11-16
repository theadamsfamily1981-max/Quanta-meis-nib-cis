"""
Distributed training utilities for TF-A-N.

Modules:
- fsdp_orchestrator: FSDP/ZeRO-3 wrapper for multi-GPU training
"""

from .fsdp_orchestrator import (
    FSDPOrchestrator,
    FSDPConfig,
    estimate_memory_savings
)

__all__ = [
    "FSDPOrchestrator",
    "FSDPConfig",
    "estimate_memory_savings"
]
