import numpy as np


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    return float((probs.argmax(axis=1) == labels).mean())


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    ez = np.exp(logits)
    return ez / ez.sum(axis=1, keepdims=True)
