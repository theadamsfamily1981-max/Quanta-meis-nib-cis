"""
Topological Data Analysis module for TFAN.
Provides persistence diagrams, persistence landscapes, and topological priors.
"""
import numpy as np
import torch
from typing import Tuple, Optional, List
import warnings

try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    warnings.warn("GUDHI not available. Nightly exact PH checks will be disabled.")

try:
    from ripser import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    warnings.warn("Ripser not available. Fast PH fallback disabled.")


class PersistenceLandscape:
    """
    Persistence Landscape (PLLay) vectorization for training.
    Converts persistence diagrams to fixed-length feature vectors.
    """
    def __init__(self, num_landscapes: int = 5, resolution: int = 100,
                 max_death: Optional[float] = None):
        """
        Args:
            num_landscapes: Number of landscape functions to compute (k)
            resolution: Number of sample points per landscape
            max_death: Maximum death time for normalization (auto if None)
        """
        self.num_landscapes = num_landscapes
        self.resolution = resolution
        self.max_death = max_death
        self.grid = None

    def fit_transform(self, pd: np.ndarray) -> np.ndarray:
        """
        Convert persistence diagram to landscape vectors.

        Args:
            pd: Persistence diagram as (birth, death) pairs, shape [N, 2]

        Returns:
            Landscape vectors, shape [num_landscapes, resolution]
        """
        if len(pd) == 0:
            return np.zeros((self.num_landscapes, self.resolution))

        # Filter infinite points and compute lifetimes
        finite_mask = np.isfinite(pd[:, 1])
        pd_finite = pd[finite_mask]

        if len(pd_finite) == 0:
            return np.zeros((self.num_landscapes, self.resolution))

        # Set up grid
        max_d = self.max_death if self.max_death else pd_finite[:, 1].max()
        min_b = pd_finite[:, 0].min()
        self.grid = np.linspace(min_b, max_d, self.resolution)

        # Compute landscape functions
        landscapes = np.zeros((self.num_landscapes, self.resolution))

        for t_idx, t in enumerate(self.grid):
            # For each point, compute tent function value at t
            tent_values = []
            for birth, death in pd_finite:
                if birth <= t <= death:
                    # Tent function: min(t - birth, death - t)
                    tent_values.append(min(t - birth, death - t))

            # Sort in descending order and take top k
            tent_values = sorted(tent_values, reverse=True)
            for k in range(min(self.num_landscapes, len(tent_values))):
                landscapes[k, t_idx] = tent_values[k]

        return landscapes

    def to_tensor(self, landscapes: np.ndarray, device: torch.device) -> torch.Tensor:
        """Convert landscape array to PyTorch tensor."""
        return torch.from_numpy(landscapes).float().to(device)


def compute_persistence_diagram(X: np.ndarray, max_dim: int = 1,
                                 engine: str = "auto") -> List[np.ndarray]:
    """
    Compute persistence diagram from point cloud.

    Args:
        X: Point cloud, shape [N, D]
        max_dim: Maximum homology dimension to compute
        engine: "gudhi", "ripser", or "auto"

    Returns:
        List of persistence diagrams, one per dimension
    """
    if engine == "auto":
        engine = "gudhi" if GUDHI_AVAILABLE else "ripser" if RIPSER_AVAILABLE else None

    if engine is None:
        raise RuntimeError("No PH engine available. Install gudhi or ripser.")

    if engine == "ripser" and RIPSER_AVAILABLE:
        result = ripser(X, maxdim=max_dim)
        # Ripser returns diagrams by dimension
        return [result['dgms'][i] for i in range(max_dim + 1)]

    elif engine == "gudhi" and GUDHI_AVAILABLE:
        # Use Rips complex from GUDHI
        rips_complex = gudhi.RipsComplex(points=X, max_edge_length=2.0)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim + 1)
        simplex_tree.compute_persistence()

        # Extract diagrams by dimension
        diagrams = []
        for dim in range(max_dim + 1):
            pairs = simplex_tree.persistence_intervals_in_dimension(dim)
            diagrams.append(pairs)

        return diagrams

    else:
        raise ValueError(f"Unknown engine: {engine}")


def wasserstein_distance(pd1: np.ndarray, pd2: np.ndarray, q: int = 2) -> float:
    """
    Compute q-Wasserstein distance between persistence diagrams.
    Simplified version using bottleneck approximation.

    Args:
        pd1, pd2: Persistence diagrams
        q: Wasserstein exponent

    Returns:
        Wasserstein distance
    """
    # Simple implementation using lifetimes
    # For production, use gudhi.wasserstein_distance
    if len(pd1) == 0 and len(pd2) == 0:
        return 0.0

    life1 = pd1[:, 1] - pd1[:, 0] if len(pd1) > 0 else np.array([])
    life2 = pd2[:, 1] - pd2[:, 0] if len(pd2) > 0 else np.array([])

    # Pad to same length
    max_len = max(len(life1), len(life2))
    life1_pad = np.pad(life1, (0, max_len - len(life1)))
    life2_pad = np.pad(life2, (0, max_len - len(life2)))

    return np.mean(np.abs(life1_pad - life2_pad) ** q) ** (1.0 / q)


def topological_kl_prior(embeddings: torch.Tensor, target_landscapes: torch.Tensor,
                         weight: float = 0.01) -> torch.Tensor:
    """
    Topological KL divergence prior for regularization.

    Args:
        embeddings: Current layer embeddings [B, N, D]
        target_landscapes: Target landscape vectors [num_landscapes, resolution]
        weight: Prior weight

    Returns:
        KL divergence loss (scalar)
    """
    # Compute persistence landscapes for current embeddings
    # This is a simplified version - in practice, compute per batch item
    batch_size = embeddings.shape[0]
    device = embeddings.device

    # Flatten for now (simplified)
    X_np = embeddings[0].detach().cpu().numpy()

    # Compute PD and landscapes
    try:
        pds = compute_persistence_diagram(X_np, max_dim=1, engine="auto")
        pl = PersistenceLandscape(num_landscapes=target_landscapes.shape[0],
                                   resolution=target_landscapes.shape[1])
        current_landscapes = pl.fit_transform(pds[1])  # Use H1
        current_landscapes_t = pl.to_tensor(current_landscapes, device)

        # KL divergence (simplified as MSE for now)
        kl = torch.nn.functional.mse_loss(current_landscapes_t, target_landscapes)

        return weight * kl
    except Exception:
        # If computation fails, return zero loss
        return torch.tensor(0.0, device=device)


@torch.no_grad()
def compute_topological_features(X: torch.Tensor, num_landscapes: int = 5,
                                  resolution: int = 100) -> torch.Tensor:
    """
    Fast topological feature extraction for landmarks.

    Args:
        X: Point cloud tensor [N, D]
        num_landscapes: Number of landscape functions
        resolution: Resolution per landscape

    Returns:
        Landscape feature tensor [num_landscapes, resolution]
    """
    X_np = X.cpu().numpy()

    try:
        pds = compute_persistence_diagram(X_np, max_dim=1, engine="auto")
        pl = PersistenceLandscape(num_landscapes=num_landscapes, resolution=resolution)
        landscapes = pl.fit_transform(pds[1])  # Use H1 (loops)
        return torch.from_numpy(landscapes).float().to(X.device)
    except Exception as e:
        warnings.warn(f"Topological feature extraction failed: {e}")
        return torch.zeros((num_landscapes, resolution), device=X.device)
