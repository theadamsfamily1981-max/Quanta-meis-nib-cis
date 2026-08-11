"""Benchmark dense attention vs TLS selective self-attention."""
from __future__ import annotations

import argparse
import time
from typing import Tuple

try:  # pragma: no cover
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

    class _DummyNN:  # type: ignore
        class Module:  # noqa: D401 - placeholder for typing
            """Placeholder type used when PyTorch is unavailable."""

            pass

        def __getattr__(self, name: str):
            raise RuntimeError("PyTorch is required for benchmark execution")

    nn = _DummyNN()  # type: ignore

from ..components.ssa import SSAConfig, SelectiveSelfAttention


def _make_data(batch: int, seq: int, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    hidden = torch.randn(batch, seq, dim, device=device)
    mask = torch.ones(batch, 1, 1, seq, device=device)
    return hidden, mask


def benchmark(args: argparse.Namespace) -> None:
    if torch is None:  # pragma: no cover
        raise ImportError("PyTorch is required for the benchmark")

    hidden, mask = _make_data(args.batch, args.sequence, args.dimension)

    dense = nn.MultiheadAttention(args.dimension, args.heads, batch_first=True).to(hidden.device)
    config = SSAConfig(embed_dim=args.dimension, num_heads=args.heads, topk_ratio=args.topk)
    tls = SelectiveSelfAttention(config).to(hidden.device)

    for _ in range(5):
        dense(hidden, hidden, hidden, key_padding_mask=None)
        tls(hidden, mask=mask)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(args.steps):
        dense(hidden, hidden, hidden, key_padding_mask=None)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    dense_time = time.time() - start

    start = time.time()
    for _ in range(args.steps):
        tls(hidden, mask=mask)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    tls_time = time.time() - start

    print(f"Dense attention: {dense_time:.4f}s for {args.steps} steps")
    print(f"TLS selective attention: {tls_time:.4f}s for {args.steps} steps")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--topk", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args(argv)
    benchmark(args)


if __name__ == "__main__":
    main()
