"""Benchmark script for tfan selective self-attention layers."""
import json
import time

import torch

from tfan.ssa import SelectiveSelfAttention
from tfan.tls import select_landmarks


def run(N: int = 8192, D: int = 1024, keep: float = 0.33) -> None:
    """Execute the benchmark and print JSON with latency and memory stats.

    Args:
        N: Sequence length.
        D: Embedding dimension.
        keep: Ratio of landmarks to retain.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(2, N, D, device=device)
    mask, _ = select_landmarks(x, keep_ratio=keep)
    attn = SelectiveSelfAttention(D).to(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    y, _ = attn(x, mask)
    _ = y
    latency_ms = (time.time() - t0) * 1000
    peak_memory = (
        torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    )

    print(
        json.dumps(
            {
                "N": N,
                "keep": keep,
                "lat_ms": latency_ms,
                "mem_peak_mb": peak_memory,
            }
        )
    )


if __name__ == "__main__":  # pragma: no cover
    run()
