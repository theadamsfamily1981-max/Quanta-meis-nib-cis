import numpy as np
from tfan.tls.selector import select_landmarks


def test_tls_shapes():
  X = np.random.randn(128, 32)
  m, s = select_landmarks(X, k=32, mode="rp", per_head=True, n_heads=4)
  assert m.shape == (4, 128)
  assert m.dtype == np.bool_
  assert s.shape == (128,)
