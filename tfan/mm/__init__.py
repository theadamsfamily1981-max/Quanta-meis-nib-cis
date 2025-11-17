"""Multi-modal processing components."""

from .ingest import ModalityAdapter, TextAdapter, AudioAdapter, VideoAdapter, ModalityStream
from .align import align_streams, validate_alignment
from .fuse import pack_and_mask, MultiModalFuser, FusedRepresentation
from .topo_gate import TopologyGate

__all__ = [
    "ModalityAdapter",
    "TextAdapter",
    "AudioAdapter",
    "VideoAdapter",
    "ModalityStream",
    "align_streams",
    "validate_alignment",
    "pack_and_mask",
    "MultiModalFuser",
    "FusedRepresentation",
    "TopologyGate",
]
