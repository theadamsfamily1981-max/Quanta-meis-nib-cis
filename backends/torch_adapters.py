from typing import Optional

from . import TORCH_AVAILABLE

if TORCH_AVAILABLE:  # pragma: no cover - runtime optional
    import torch
    import torch.nn as nn

    class LoRALinear(nn.Module):
        def __init__(self, in_features: int, out_features: int, r: int, alpha: float = 1.0, bias: bool = False):
            super().__init__()
            self.base = nn.Linear(in_features, out_features, bias=bias)
            self.r = r
            self.scaling = alpha / max(r, 1)
            self.A = nn.Parameter(torch.zeros(r, in_features))
            self.B = nn.Parameter(torch.zeros(out_features, r))
            nn.init.kaiming_uniform_(self.A, a=5**0.5)
            nn.init.zeros_(self.B)

        def forward(self, x):
            delta = (x @ self.A.t()) @ self.B.t()
            return self.base(x) + self.scaling * delta

    class DoRALinear(nn.Module):
        def __init__(self, in_features: int, out_features: int, r: int, bias: bool = False):
            super().__init__()
            self.base = nn.Linear(in_features, out_features, bias=bias)
            self.U = nn.Parameter(torch.randn(r, in_features) * 0.02)
            self.V = nn.Parameter(torch.randn(out_features, r) * 0.02)
            self.gain = nn.Parameter(torch.ones(out_features))

        def forward(self, x):
            approx = (x @ self.U.t()) @ self.V.t()
            return self.base(x) + self.gain * approx
else:
    # Stubs to keep import-time errors away in CI
    class LoRALinear:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("Torch not available; install torch to use LoRALinear")

    class DoRALinear:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("Torch not available; install torch to use DoRALinear")
