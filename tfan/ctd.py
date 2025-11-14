"""
CTD (Curvature-Topology Detection) with hyperbolic geometry support.
Auto-enables Poincaré embeddings when data exhibits tree-like structure.
Gate: NDCG@K +5% vs Euclidean or auto-disable.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

try:
    import geoopt
    GEOOPT_AVAILABLE = True
except ImportError:
    GEOOPT_AVAILABLE = False


class TreeLikenessDetector:
    """Detect tree-like structure in embeddings."""

    @staticmethod
    def compute_tree_score(embeddings: torch.Tensor, k: int = 10) -> float:
        """
        Compute tree-likeness score using distance distribution.
        Tree-like data has low δ-hyperbolicity.

        Args:
            embeddings: [N, D] tensor
            k: number of neighbors to check

        Returns:
            Tree score in [0, 1], higher = more tree-like
        """
        N = embeddings.shape[0]
        if N < 4:
            return 0.0

        # Compute pairwise distances
        dists = torch.cdist(embeddings, embeddings)

        # Sample quadruples and compute Gromov product
        # δ = max over quadruples of (xy + zw - max(xz+yw, xw+yz))/2
        num_samples = min(100, N // 4)
        hyperbolicity_samples = []

        for _ in range(num_samples):
            # Random quadruple
            idx = torch.randperm(N)[:4]
            x, y, z, w = idx

            xy = dists[x, y].item()
            xz = dists[x, z].item()
            xw = dists[x, w].item()
            yz = dists[y, z].item()
            yw = dists[y, w].item()
            zw = dists[z, w].item()

            # Gromov product
            delta = (xy + zw - max(xz + yw, xw + yz)) / 2.0
            hyperbolicity_samples.append(abs(delta))

        # Low hyperbolicity = tree-like
        avg_delta = np.mean(hyperbolicity_samples)
        # Normalize to [0, 1]
        tree_score = np.exp(-avg_delta)

        return tree_score

    @staticmethod
    def compute_spectral_dim(embeddings: torch.Tensor) -> float:
        """
        Estimate intrinsic dimensionality via spectral methods.

        Returns:
            Estimated dimension
        """
        # Use PCA to estimate intrinsic dim
        centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        cov = torch.mm(centered.T, centered) / (embeddings.shape[0] - 1)
        eigenvalues = torch.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues.flip(0)  # Descending order

        # Count eigenvalues that explain 95% variance
        total_var = eigenvalues.sum()
        cumsum = torch.cumsum(eigenvalues, dim=0)
        dim_95 = (cumsum / total_var < 0.95).sum().item() + 1

        return float(dim_95)


class PoincareEmbedding(nn.Module):
    """Poincaré ball embedding layer."""

    def __init__(self, num_embeddings: int, embedding_dim: int, c: float = 1.0):
        super().__init__()

        if not GEOOPT_AVAILABLE:
            raise ImportError("geoopt required for Poincaré embeddings. Install with: pip install geoopt")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.c = c  # Curvature

        # Poincaré ball manifold
        self.ball = geoopt.PoincareBall(c=c)

        # Initialize embeddings on the manifold
        self.weight = geoopt.ManifoldParameter(
            torch.randn(num_embeddings, embedding_dim) * 0.01,
            manifold=self.ball
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Embed indices into Poincaré ball.

        Args:
            indices: [B] tensor of indices

        Returns:
            Embeddings [B, D] in Poincaré ball
        """
        return self.weight[indices]

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Poincaré distance between points."""
        return self.ball.dist(x, y)


class HyperbolicLayer(nn.Module):
    """
    Hyperbolic neural network layer using Poincaré ball model.
    Auto-enables when tree-likeness detected.
    """

    def __init__(self, in_features: int, out_features: int, c: float = 1.0,
                 use_euclidean_fallback: bool = True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.use_euclidean_fallback = use_euclidean_fallback
        self.use_hyperbolic = GEOOPT_AVAILABLE

        if self.use_hyperbolic:
            self.ball = geoopt.PoincareBall(c=c)
            # Mobius linear layer
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
            self.bias = geoopt.ManifoldParameter(
                torch.zeros(out_features, 1),
                manifold=self.ball
            )
        else:
            # Euclidean fallback
            self.linear = nn.Linear(in_features, out_features)

    def mobius_matvec(self, m: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Möbius matrix-vector multiplication."""
        if not self.use_hyperbolic:
            return self.linear(x)

        # Project to tangent space at origin
        x_tan = self.ball.logmap0(x)

        # Apply linear transformation
        mx_tan = torch.mm(m, x_tan.T).T

        # Map back to manifold
        mx = self.ball.expmap0(mx_tan)

        return mx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, in_features] input (on manifold if hyperbolic)

        Returns:
            [B, out_features] output
        """
        if not self.use_hyperbolic:
            return self.linear(x)

        # Möbius transformation
        out = self.mobius_matvec(self.weight, x)

        # Möbius addition with bias
        out = self.ball.mobius_add(out, self.bias.squeeze(1))

        return out


class CTDGate:
    """
    CTD gate that decides whether to enable hyperbolic geometry.
    Uses tree-likeness heuristics and performance metrics.
    """

    def __init__(self, ndcg_threshold: float = 0.05, enable_threshold: float = 0.7):
        self.ndcg_threshold = ndcg_threshold
        self.enable_threshold = enable_threshold

    def should_enable_hyperbolic(self, embeddings: torch.Tensor,
                                 euclidean_ndcg: Optional[float] = None,
                                 hyperbolic_ndcg: Optional[float] = None) -> Tuple[bool, dict]:
        """
        Decide whether to enable hyperbolic embeddings.

        Args:
            embeddings: Current embeddings to analyze
            euclidean_ndcg: NDCG score with Euclidean embeddings
            hyperbolic_ndcg: NDCG score with hyperbolic embeddings

        Returns:
            (should_enable, metrics_dict)
        """
        detector = TreeLikenessDetector()

        # Compute heuristics
        tree_score = detector.compute_tree_score(embeddings)
        spectral_dim = detector.compute_spectral_dim(embeddings)

        metrics = {
            "tree_score": tree_score,
            "spectral_dim": spectral_dim,
        }

        # Decision logic
        enable = tree_score > self.enable_threshold

        # If we have NDCG comparisons, use them
        if euclidean_ndcg is not None and hyperbolic_ndcg is not None:
            ndcg_gain = hyperbolic_ndcg - euclidean_ndcg
            metrics["euclidean_ndcg"] = euclidean_ndcg
            metrics["hyperbolic_ndcg"] = hyperbolic_ndcg
            metrics["ndcg_gain"] = ndcg_gain

            # Enable if gain > threshold
            enable = enable and (ndcg_gain > self.ndcg_threshold)

        metrics["enable_hyperbolic"] = enable

        return enable, metrics


def ndcg_at_k(relevance_scores: np.ndarray, k: int = 10) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at K.

    Args:
        relevance_scores: Relevance scores in ranked order
        k: Cutoff position

    Returns:
        NDCG@K score
    """
    relevance_scores = relevance_scores[:k]

    if len(relevance_scores) == 0:
        return 0.0

    # DCG
    dcg = relevance_scores[0]
    for i in range(1, len(relevance_scores)):
        dcg += relevance_scores[i] / np.log2(i + 2)

    # Ideal DCG
    ideal_scores = np.sort(relevance_scores)[::-1]
    idcg = ideal_scores[0]
    for i in range(1, len(ideal_scores)):
        idcg += ideal_scores[i] / np.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg
