from __future__ import annotations
import os
import csv
import numpy as np

class MELDLoader:
    def __init__(self, root: str | None = None):
        self.root = root or os.environ.get('MELD_ROOT', 'data/MELD')
        self.ok = os.path.exists(self.root)

    def sample(self, n=64, d=16, k=4):
        g = np.random.default_rng(0)
        x = g.standard_normal((n, d)).astype(np.float32)
        y = g.integers(0, k, size=(n,), dtype=np.int64)
        return x, y


class IEMOCAPLoader:
    def __init__(self, root: str | None = None):
        self.root = root or os.environ.get('IEMOCAP_ROOT', 'data/IEMOCAP')
        self.ok = os.path.exists(self.root)

    def sample(self, n=64, d=16, k=4):
        g = np.random.default_rng(1)
        x = g.standard_normal((n, d)).astype(np.float32)
        y = g.integers(0, k, size=(n,), dtype=np.int64)
        return x, y
