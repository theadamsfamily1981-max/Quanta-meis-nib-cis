"""Tests for topology module."""
import pytest
import numpy as np
import torch

from tfan.topo import (
    PersistenceLandscape,
    compute_persistence_diagram,
    wasserstein_distance,
    GUDHI_AVAILABLE,
    RIPSER_AVAILABLE
)


def test_persistence_landscape():
    """Test persistence landscape computation."""
    # Create simple persistence diagram (birth, death pairs)
    pd = np.array([
        [0.0, 0.5],
        [0.1, 0.8],
        [0.2, 0.6],
    ])

    pl = PersistenceLandscape(num_landscapes=3, resolution=50)
    landscapes = pl.fit_transform(pd)

    assert landscapes.shape == (3, 50)
    assert np.all(landscapes >= 0)  # Landscapes should be non-negative
    assert landscapes[0].max() > 0  # First landscape should have some signal


def test_persistence_landscape_empty():
    """Test persistence landscape with empty diagram."""
    pd = np.array([]).reshape(0, 2)

    pl = PersistenceLandscape(num_landscapes=3, resolution=50)
    landscapes = pl.fit_transform(pd)

    assert landscapes.shape == (3, 50)
    assert np.all(landscapes == 0)  # Should be all zeros


@pytest.mark.skipif(not (GUDHI_AVAILABLE or RIPSER_AVAILABLE),
                   reason="No PH engine available")
def test_compute_persistence_diagram():
    """Test PH computation on simple point cloud."""
    # Create circle point cloud
    theta = np.linspace(0, 2 * np.pi, 50)
    X = np.column_stack([np.cos(theta), np.sin(theta)])

    diagrams = compute_persistence_diagram(X, max_dim=1)

    assert len(diagrams) == 2  # H0 and H1
    assert len(diagrams[1]) > 0  # Should detect the loop (H1)


def test_wasserstein_distance():
    """Test Wasserstein distance computation."""
    pd1 = np.array([
        [0.0, 0.5],
        [0.1, 0.6],
    ])

    pd2 = np.array([
        [0.0, 0.5],
        [0.1, 0.65],  # Slightly different
    ])

    dist = wasserstein_distance(pd1, pd2, q=2)

    assert dist >= 0  # Distance should be non-negative
    assert dist < 0.1  # Should be small since diagrams are similar


def test_wasserstein_distance_identical():
    """Test Wasserstein distance of identical diagrams."""
    pd = np.array([
        [0.0, 0.5],
        [0.1, 0.6],
    ])

    dist = wasserstein_distance(pd, pd, q=2)

    assert dist < 1e-6  # Should be essentially zero


def test_landscape_to_tensor():
    """Test conversion to PyTorch tensor."""
    pd = np.array([
        [0.0, 0.5],
        [0.1, 0.8],
    ])

    pl = PersistenceLandscape(num_landscapes=3, resolution=50)
    landscapes = pl.fit_transform(pd)
    tensor = pl.to_tensor(landscapes, device=torch.device("cpu"))

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 50)
    assert tensor.device.type == "cpu"
