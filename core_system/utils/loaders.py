"""
Placeholder dataloaders.
"""
import torch

def load_synthetic_data(size=32, dim=10):
    return torch.randn(size, dim), torch.randn(size, dim)
