"""Multi-Modal integration package."""

from .ingest import (
    TextAdapter,
    AudioAdapter,
    VideoAdapter,
    IMUAdapter,
    create_adapter
)

from .align import align_streams, TTWAligner, TTWConfig, AlignmentResult

__all__ = [
    'TextAdapter',
    'AudioAdapter',
    'VideoAdapter',
    'IMUAdapter',
    'create_adapter',
    'align_streams',
    'TTWAligner',
    'TTWConfig',
    'AlignmentResult'
]
