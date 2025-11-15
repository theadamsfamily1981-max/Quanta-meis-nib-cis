"""
TF-A-N: Transformer with Formal Alignment and Neuromodulation

Long-sequence, multi-modal inference at production latencies with
provable structural fidelity and homeostatic stability.
"""

__version__ = "0.1.0"

from .config import TFANConfig
from .topo import TopologyRegularizer
from .attention import SparseAttention, TLSLandmarkSelector
from .pgu import ProofGatedUpdater
from .trainer import TFANTrainer

__all__ = [
    "TFANConfig",
    "TopologyRegularizer",
    "SparseAttention",
    "TLSLandmarkSelector",
    "ProofGatedUpdater",
    "TFANTrainer",
]
