import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform


def mst_lifetime_scores(X):
  """
  Build MST on Euclidean distances, score nodes by incident edge 'lifetimes':
  sum of top-2 longest incident edges = saliency proxy for persistence.
  """
  D = squareform(pdist(X, metric="euclidean"))
  G = nx.Graph()
  N = X.shape[0]
  for i in range(N):
    for j in range(i + 1, N):
      G.add_edge(i, j, weight=D[i, j])
  T = nx.minimum_spanning_tree(G, weight='weight')
  # lifetime: nodes with long incident edges are salient junctions/bridges
  scores = np.zeros(N, dtype=float)
  for n in T.nodes():
    inc = [T[n][nb]['weight'] for nb in T.neighbors(n)]
    inc_sorted = sorted(inc, reverse=True)
    if len(inc_sorted) == 0:
      scores[n] = 0.0
    elif len(inc_sorted) == 1:
      scores[n] = inc_sorted[0]
    else:
      scores[n] = inc_sorted[0] + 0.5 * inc_sorted[1]
  return scores


def select_mst_lifetime(X, k):
  scores = mst_lifetime_scores(X)
  idx = np.argsort(scores)[-k:]
  return np.sort(idx), scores
