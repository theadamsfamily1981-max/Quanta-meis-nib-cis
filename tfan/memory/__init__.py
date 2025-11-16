"""
TF-A-N CXL/UMA Memory Tiering

This module provides CXL-aware memory tiering for extreme long-context inference:
- CXLPager: Multi-tier KV cache with NVMe/CXL as cold storage
- BloomPrefetcher: Bloom filter-based access pattern prediction
- UMAOptimizer: Unified memory architecture optimizations

Hard gates:
- 128k context without OOM on 24GB GPU
- ≤8% tokens/s penalty vs in-memory cache
- Cache hit-rate ≥90%
- Prefetch accuracy ≥80%
"""

from .cxl_pager import CXLPager, CXLPageConfig
from .bloom import BloomPrefetcher, BloomConfig

__all__ = ['CXLPager', 'CXLPageConfig', 'BloomPrefetcher', 'BloomConfig']
