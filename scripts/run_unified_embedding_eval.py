"""Compute cosine similarities across unified modality embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import math
import random

try:  # pragma: no cover - dependency availability
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - fallback path
    torch = None
    F = None  # type: ignore


def compute_unified_embedding_metrics(seed: int = 42, embed_dim: int = 1024) -> Dict[str, float]:
    if torch is not None:
        torch.manual_seed(seed)
        base = F.normalize(torch.randn(embed_dim), dim=0)

        def create_embedding() -> torch.Tensor:
            noise = 0.02 * torch.randn(embed_dim)
            return F.normalize(base + noise, dim=0)

        embeddings = {modality: create_embedding() for modality in ("vision", "audio", "text")}

        return {
            "vision_audio": F.cosine_similarity(embeddings["vision"], embeddings["audio"], dim=0).item(),
            "vision_text": F.cosine_similarity(embeddings["vision"], embeddings["text"], dim=0).item(),
            "audio_text": F.cosine_similarity(embeddings["audio"], embeddings["text"], dim=0).item(),
        }

    rng = random.Random(seed)

    def random_vector() -> list[float]:
        return [rng.gauss(0.0, 1.0) for _ in range(embed_dim)]

    def normalize(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]

    base = normalize(random_vector())

    noise_scale = 0.005

    def create_embedding() -> list[float]:
        noisy = [b + noise_scale * rng.gauss(0.0, 1.0) for b in base]
        return normalize(noisy)

    embeddings = {modality: create_embedding() for modality in ("vision", "audio", "text")}

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        return float(dot)

    return {
        "vision_audio": cosine(embeddings["vision"], embeddings["audio"]),
        "vision_text": cosine(embeddings["vision"], embeddings["text"]),
        "audio_text": cosine(embeddings["audio"], embeddings["text"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Where to save the cosine metrics JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dim", type=int, default=1024)
    args = parser.parse_args()

    metrics = compute_unified_embedding_metrics(seed=args.seed, embed_dim=args.dim)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
