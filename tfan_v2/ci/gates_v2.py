"""CI adapter exposing a minimal smoke test harness."""
from __future__ import annotations

import numpy as np

from ..components.topology import summarise_points

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


def run() -> None:
    """Execute a quick smoke test used by the CI workflow."""

    points = np.random.RandomState(0).randn(32, 3)
    signature = summarise_points(points)
    if signature.lifetimes.size == 0:
        raise RuntimeError("Topology summary failed")

    if torch is not None:
        from ..components.ssa import SSAConfig, SelectiveSelfAttention

        config = SSAConfig(embed_dim=16, num_heads=4, topk_ratio=0.5)
        attn = SelectiveSelfAttention(config)
        dummy = torch.randn(2, 10, 16)
        attn(dummy)

    print("CI gate executed successfully")


if __name__ == "__main__":
    run()
