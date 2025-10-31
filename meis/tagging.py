import numpy as np

# Synaptic Tagging & Capture (STC)-like tagging: simple importance + time decay

class Tagger:
    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.tags = None

    def update(self, grad_like: np.ndarray):
        if self.tags is None:
            self.tags = np.abs(grad_like)
        else:
            self.tags = self.decay * self.tags + (1 - self.decay) * np.abs(grad_like)
        return self.tags


def fisher_salience(grads: np.ndarray) -> np.ndarray:
    # Proxy: squared grads average across batch
    return (grads ** 2).mean(axis=0)


def svd_consolidate_matrix(W: np.ndarray, tags: np.ndarray, k: int) -> np.ndarray:
    # Weighted SVD keeping top-k singular modes; tag weights emphasize salient rows
    weight = np.clip(tags.reshape(-1, 1), 0.1, None)
    Ww = weight * W
    U, S, Vt = np.linalg.svd(Ww, full_matrices=False)
    S_trunc = np.zeros_like(S)
    S_trunc[:k] = S[:k]
    return (U * S_trunc) @ Vt
