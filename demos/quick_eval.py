import json, time
import numpy as np
from common.pad import PAD
from common.rank import rank_schedule
from quanta.adapters import LoRA, DoRA, rank_decay_schedule, consolidate_svd
from cis.pad_gate import PADGate, PADThresholds
from cis.datasets import EmotionToy
from cis.metrics import ECE, brier_score


if __name__ == "__main__":
    ds = EmotionToy()
    W = np.zeros((4, 16), dtype=np.float32)
    lora = LoRA(r=8, alpha=4.0).init(16, 4)
    x, y, pad = ds.batch(64)
    yhat = lora.forward(x, W)
    probs = np.clip(np.exp(yhat) / np.exp(yhat).sum(axis=1, keepdims=True), 1e-6, 1)
    ece = ECE(10)(probs, y)
    bs = brier_score(probs, y, 4)
    pad_gate = PADGate(PADThresholds(0,0,0))
    open_ratio = float(np.mean([pad_gate.open(tuple(p), 1.0) for p in pad]))

    out = {"ece": ece, "brier": bs, "open_ratio": open_ratio, "rank_sched": rank_schedule(64, 9)}
    print(json.dumps(out, indent=2))
