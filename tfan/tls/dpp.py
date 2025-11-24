import numpy as np


def rbf_kernel(X, gamma=None):
  # X: [N, d]
  if gamma is None:
    # median heuristic
    dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    med = np.median(dists)
    gamma = 1.0 / (2.0 * (med**2 + 1e-12))
  sq = np.sum(X**2, axis=1, keepdims=True)
  K = -2 * np.dot(X, X.T) + sq + sq.T
  return np.exp(-gamma * np.maximum(K, 0.0))


def k_dpp_rbf(X, k, gamma=None, seed=0):
  """
  Lightweight k-DPP sampler (Nystrom-ish eigen stub).
  Returns indices of size k emphasizing diversity.
  """
  rng = np.random.default_rng(seed)
  K = rbf_kernel(X, gamma)
  # Add small jitter for numerical stability
  K = (K + K.T) / 2.0 + 1e-8 * np.eye(K.shape[0])
  # Eigen decomp (stub; fine for k up to few hundreds)
  vals, vecs = np.linalg.eigh(K)
  vals = np.clip(vals, 0.0, None)
  # Sample eigenvectors proportional to lambda/(1+lambda)
  probs = vals / (vals + 1.0)
  chosen = rng.random(len(vals)) < probs
  V = vecs[:, chosen]
  if V.size == 0:
    # fallback to farthest-point
    return farthest_point(X, k)
  Y = []
  # Greedy k-DPP
  for _ in range(min(k, V.shape[1])):
    # select item proportional to row norm in V
    scores = np.sum(V**2, axis=1)
    i = int(np.argmax(scores))
    Y.append(i)
    vi = V[i, :].copy()
    if np.allclose(vi, 0):
      break
    # Gram-Schmidt deflation
    vi = vi / np.linalg.norm(vi)
    V = V - np.outer(np.dot(V, vi), vi)
  # If not enough, pad with FPS
  if len(Y) < k:
    pad = farthest_point(X, k - len(Y), init=Y)
    Y += pad
  return np.array(sorted(set(Y)))[:k]


def farthest_point(X, k, init=None):
  N = X.shape[0]
  chosen = list(init) if init else [int(np.random.randint(N))]
  d = np.linalg.norm(X - X[chosen[0]], axis=1)
  for _ in range(k - len(chosen)):
    i = int(np.argmax(d))
    chosen.append(i)
    d = np.minimum(d, np.linalg.norm(X - X[i], axis=1))
  return chosen
