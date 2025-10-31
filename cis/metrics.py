import numpy as np

class ECE:
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def __call__(self, probs: np.ndarray, labels: np.ndarray) -> float:
        bins = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        conf = probs.max(axis=1)
        preds = probs.argmax(axis=1)
        correct = (preds == labels).astype(np.float32)
        for i in range(self.n_bins):
            m = (conf >= bins[i]) & (conf < bins[i + 1])
            if m.sum() == 0:
                continue
            acc = correct[m].mean()
            c = conf[m].mean()
            ece += (m.mean()) * abs(acc - c)
        return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    onehot = np.eye(n_classes, dtype=np.float32)[labels]
    return float(((probs - onehot) ** 2).mean())
