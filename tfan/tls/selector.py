import numpy as np
from .dpp import k_dpp_rbf
from .mst import select_mst_lifetime
from .rp import randomized_projection_kmeans


def select_landmarks(X, k, mode="rp", alpha_persistence=0.7, per_head=False, n_heads=8, seed=0):
  """
  X: np.ndarray [N, d] token embeddings (CPU np array expected here)
  mode: 'dpp' | 'mst' | 'rp'
  alpha_persistence: weight for 'persistence' score in hybridization (kept for compat)
  per_head: if True, create per-head masks by jittering selection
  """
  N = X.shape[0]
  k = max(1, min(k, N))
  if mode == "dpp":
    idx = k_dpp_rbf(X, k, seed=seed)
    scores = np.zeros(N)
    scores[idx] = 1.0
  elif mode == "mst":
    idx, scores = select_mst_lifetime(X, k)
  else:
    idx = randomized_projection_kmeans(X, k, seed=seed)
    scores = np.zeros(N)
    scores[idx] = 1.0

  mask = np.zeros(N, dtype=bool)
  mask[idx] = True
  if not per_head:
    return mask[None, :], scores  # [1, N]
  # Per-head landmark masks: small stochastic variations
  rng = np.random.default_rng(seed)
  masks = []
  base = np.where(mask)[0]
  for h in range(n_heads):
    # random drop/add 5% of landmarks to specialize per head
    head = base.copy()
    if len(head) > 0:
      drop_n = max(0, int(0.05 * len(head)))
      if drop_n > 0:
        drop = rng.choice(head, size=drop_n, replace=False)
        head = np.setdiff1d(head, drop)
    # add a few nearby tokens
    add_n = max(0, int(0.05 * k))
    add = rng.choice(np.setdiff1d(np.arange(N), head), size=add_n, replace=False)
    head = np.unique(np.concatenate([head, add]))
    m = np.zeros(N, dtype=bool)
    m[head] = True
    masks.append(m)
  return np.stack(masks, axis=0), scores
