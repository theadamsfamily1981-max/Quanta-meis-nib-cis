"""Utility functions for selecting landmark tokens from embeddings."""
from __future__ import annotations

from typing import Dict, Tuple

import torch


def select_landmarks(
    x: torch.Tensor,
    keep_ratio: float = 0.33,
    strategy: str = "hybrid",
    knn: int = 32,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Select landmark tokens from a batch of embeddings.

    Parameters
    ----------
    x:
        Tensor of shape ``[B, N, D]`` containing a batch of token embeddings.
    keep_ratio:
        Fraction of tokens to keep per sequence. At least one token is always
        selected.
    strategy:
        Name of the selection strategy. Currently unused but kept for
        compatibility with downstream configuration.
    knn:
        Placeholder parameter mirroring the original signature. It is returned
        in the metrics for potential logging.

    Returns
    -------
    Tuple[torch.Tensor, Dict[str, float]]
        * A boolean tensor mask of shape ``[B, N]`` indicating which tokens
          have been selected as landmarks.
        * A metrics dictionary that records selection metadata.
    """

    if x.ndim != 3:
        raise ValueError("Expected input tensor to have shape [B, N, D]")

    batch, num_tokens, _ = x.shape

    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in the interval (0, 1]")

    k = max(1, int(round(num_tokens * keep_ratio)))

    # Fast proxy scores: L2 distance to the batch centroid as a geometry surrogate
    centroid = x.mean(dim=1, keepdim=True)
    geometry_scores = (x - centroid).pow(2).sum(dim=-1).sqrt()

    # Select the ``k`` tokens with the largest geometry scores.
    topk_scores, topk_indices = torch.topk(geometry_scores, k=k, dim=1)

    landmark_mask = torch.zeros(batch, num_tokens, dtype=torch.bool, device=x.device)
    landmark_mask.scatter_(1, topk_indices, True)

    metrics = {
        "k": float(k),
        "keep_ratio": float(keep_ratio),
        "strategy": strategy,
        "knn": float(knn),
        "mean_score": topk_scores.mean().item() if topk_scores.numel() else 0.0,
    }

    return landmark_mask, metrics
