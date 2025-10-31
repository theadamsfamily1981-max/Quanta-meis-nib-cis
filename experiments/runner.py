import json
import os
import numpy as np
from pathlib import Path

from cis.datasets import EmotionToy
from cis.metrics import ECE, brier_score
from backends import TORCH_AVAILABLE

if TORCH_AVAILABLE and os.environ.get("QMNC_BACKEND") == "torch":
    from backends.torch_adapters import LoRALinear as TorchLoRA  # type: ignore


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def run(config):
    rng = np.random.default_rng(config.get("seed", 0))
    ds = EmotionToy(n=512, d=16, n_classes=4, seed=config.get("seed", 0))
    x, y, pad = ds.batch(128)

    # Model: NumPy LoRA-like via random matrix, or Torch if requested
    if TORCH_AVAILABLE and config.get("backend") == "torch":
        import torch
        layer = TorchLoRA(16, 4, r=8, alpha=4.0)
        with torch.no_grad():
            out = layer.base(torch.from_numpy(x))
            yhat = out.detach().numpy()
    else:
        W = rng.standard_normal((4, 16)).astype(np.float32) * 0.1
        yhat = x @ W.T

    probs = softmax(yhat)
    ece_val = ECE(config.get("ece_bins", 10))(probs, y)
    brier = brier_score(probs, y, 4)
    acc = float((probs.argmax(1) == y).mean())

    return {
        "acc": acc,
        "ece": ece_val,
        "brier": brier
    }


if __name__ == "__main__":
    cfg_path = Path(__file__).parent / "configs" / "default.json"
    cfg = json.loads(cfg_path.read_text())
    res = run(cfg)
    print(json.dumps(res, indent=2))
