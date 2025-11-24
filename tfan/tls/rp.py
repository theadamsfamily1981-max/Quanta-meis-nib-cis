import numpy as np
from sklearn.cluster import KMeans


def randomized_projection_kmeans(X, k, rp_dim=10, seed=0, n_init=5, iters=100):
  rng = np.random.default_rng(seed)
  # Random Gaussian projection to rp_dim
  R = rng.normal(size=(X.shape[1], rp_dim)) / np.sqrt(rp_dim)
  Z = X @ R
  km = KMeans(n_clusters=k, n_init=n_init, max_iter=iters, random_state=seed)
  km.fit(Z)
  # pick nearest original points to cluster centers in projected space
  centers = km.cluster_centers_
  idx = []
  for c in centers:
    d = np.linalg.norm(Z - c[None, :], axis=1)
    idx.append(int(np.argmin(d)))
  idx = np.array(sorted(set(idx)))
  # If duplicates reduced k, pad with farthest points in Z
  while len(idx) < k:
    dists = np.min([np.linalg.norm(Z - Z[i], axis=1) for i in idx], axis=0)
    nxt = int(np.argmax(dists))
    if nxt in idx:
      break
    idx = np.append(idx, nxt)
    idx = np.unique(idx)
  return idx[:k]
