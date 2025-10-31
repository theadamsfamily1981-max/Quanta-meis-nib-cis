from experiments.harness import softmax
import numpy as np


def test_softmax_rows_sum_to_one():
    x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    p = softmax(x)
    assert np.allclose(p.sum(axis=1), 1.0)
