from experiments.ood import energy_score, selective_accuracy
import numpy as np


def test_energy_shape_and_direction():
    x = np.array([[0.0, 0.0], [3.0, -3.0]], dtype=np.float32)
    e = energy_score(x)
    assert e.shape == (2,)
    # Higher logit spread should lead to lower (more negative) energy for the confident sample
    assert e[1] < e[0]


def test_selective_accuracy_bounds():
    probs = np.array([[0.6, 0.4], [0.9, 0.1], [0.55, 0.45]], dtype=np.float32)
    y = np.array([0, 0, 1])
    acc = selective_accuracy(probs, y, 0.66)
    assert 0.0 <= acc <= 1.0
