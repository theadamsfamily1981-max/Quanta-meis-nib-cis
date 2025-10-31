import numpy as np

class EmotionToy:
    def __init__(self, n=256, d=16, n_classes=4, seed=0):
        g = np.random.default_rng(seed)
        self.x = g.standard_normal((n, d)).astype(np.float32)
        # Simple PAD mapping to 4 classes
        pad = g.uniform(-1, 1, size=(n, 3)).astype(np.float32)
        self.pad = pad
        # quadrant labels by pleasure vs arousal
        self.y = ((pad[:,0] > 0).astype(int) * 2 + (pad[:,1] > 0).astype(int)).astype(int)
        self.n_classes = n_classes

    def batch(self, bs=32):
        g = np.random.default_rng()
        idx = g.choice(len(self.x), size=bs, replace=False)
        return self.x[idx], self.y[idx], self.pad[idx]
