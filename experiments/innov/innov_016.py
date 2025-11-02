from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class Innov016Config:
    coverage_target: float = 0.8
    seed: int = 0


def run_innov_016(cfg: Innov016Config | None = None) -> Dict[str, Any]:
    """
    Causal gating (SCM) stub: simulate confounding and evaluate counterfactual-aware thresholding.
    Returns coverage, selective accuracy, and counterfactual stability on synthetic signals.
    ""
    cfg = cfg or Innov016Config()
    rng = np.random.default_rng(cfg.seed)

    n = 512
    # latent cause C, spurious S, noise N
    C = rng.normal(size=(n,))
    S = 0.8 * C + 0.6 * rng.normal(size=(n,))
    N = rng.normal(size=(n,))

    # label depends on C (not S)
    y = (C + 0.2 * N > 0).astype(int)

    # predictor score that spuriously mixes S
    score_spurious = 0.7 * S + 0.3 * C + 0.2 * N
    # causal-adjusted score (backdoor: regress out S)
    score_causal = score_spurious - 0.6 * S

    def selective_metrics(score, label, coverage=cfg.coverage_target):
        thr = np.quantile(score, 1 - coverage)
        sel = score >= thr
        acc = float((label[sel] == (score[sel] > np.median(score[sel]))).mean()) if sel.any() else 0.0
        return acc, float(sel.mean())

    acc_s, cov_s = selective_metrics(score_spurious, y)
    acc_c, cov_c = selective_metrics(score_causal, y)

    # counterfactual: intervene S := -S
    score_cf = 0.7 * (-S) + 0.3 * C + 0.2 * N - 0.6 * (-S)
    acc_cf, cov_cf = selective_metrics(score_cf, y)

    stability = 1.0 - abs(acc_c - acc_cf)

    return {
        "suite_id": "INNOV-016",
        "selective_acc_spurious": round(acc_s, 4),
        "selective_acc_causal": round(acc_c, 4),
        "coverage": round(cov_c, 4),
        "counterfactual_stability": round(stability, 4),
    }
