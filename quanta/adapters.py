import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class LoRA:
    r: int
    alpha: float = 1.0
    scale: float = 1.0

    def init(self, in_features: int, out_features: int, rng: np.random.Generator | None = None):
        g = rng or np.random.default_rng()
        self.A = (g.standard_normal((self.r, in_features)) * 0.02).astype(np.float32)
        self.B = (g.standard_normal((out_features, self.r)) * 0.02).astype(np.float32)
        return self

    def forward(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        # y = xW^T + (alpha/r) * x A^T B^T
        delta = (self.alpha / max(self.r, 1)) * (x @ self.A.T) @ self.B.T
        return x @ W.T + self.scale * delta


@dataclass
class DoRA:
    r: int
    def init(self, in_features: int, out_features: int, rng: np.random.Generator | None = None):
        g = rng or np.random.default_rng()
        self.U = (g.standard_normal((self.r, in_features)) * 0.02).astype(np.float32)
        self.V = (g.standard_normal((out_features, self.r)) * 0.02).astype(np.float32)
        self.gain = np.ones((out_features, 1), dtype=np.float32)
        return self

    def forward(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        # Decompose W ≈ VU, modulate by per-row gain
        approx = (x @ self.U.T) @ self.V.T
        return (x @ W.T) + self.gain.T * approx


def rank_decay_schedule(initial: int, steps: int, decay_points: Tuple[float, float] = (0.33, 0.66)) -> list[int]:
    seq = []
    for i in range(steps):
        t = i / max(steps - 1, 1)
        if t < decay_points[0]:
            seq.append(max(1, initial))
        elif t < decay_points[1]:
            seq.append(max(1, initial // 2))
        else:
            seq.append(max(1, max(1, initial // 8)))
    return seq


def consolidate_svd(W: np.ndarray, k: int) -> np.ndarray:
    # Sleep-like consolidation: keep top-k singular modes
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    S_trunc = np.zeros_like(S)
    S_trunc[:k] = S[:k]
    return (U * S_trunc) @ Vt
