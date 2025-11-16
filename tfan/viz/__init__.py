"""
Visualization and streaming infrastructure for TF-A-N.

Provides real-time WebSocket streaming of:
- Persistence diagrams (PD)
- Attention sparsity patterns
- Topological landmark selection
- FDT/EPR-CV metrics
"""

from .stream import VizStream
from .encoders import (
    encode_pd,
    encode_attention_matrix,
    encode_sparsity_metrics,
    encode_fdt_state
)

__all__ = [
    'VizStream',
    'encode_pd',
    'encode_attention_matrix',
    'encode_sparsity_metrics',
    'encode_fdt_state',
]
