import json
import numpy as np
from pathlib import Path

from cis.metrics import ECE, brier_score
from experiments.harness import softmax
from experiments.ood import energy_score, mahalanobis_scores, risk_coverage_curve, selective_accuracy


if __name__ == "__main__":
    # Synthetic logits/features for demo; integrate with runner outputs in next PR
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((512, 4)).astype(np.float32)
    probs = softmax(logits)
    labels = rng.integers(0, 4, size=(512,), dtype=np.int64)

    # Energy OOD (thresholding would be applied in practice)
    energy = energy_score(logits)

    # Mahalanobis: pretend features = logits, estimate means/cov from a split
    feats = logits
    means = np.stack([feats[labels == c].mean(axis=0) for c in range(4)], axis=0)
    cov = np.cov(feats.T) + 1e-4 * np.eye(feats.shape[1])
    maha = mahalanobis_scores(feats, means, cov).min(axis=1)

    # Calibration
    ece = ECE(10)(probs, labels)
    brier = brier_score(probs, labels, 4)

    # Risk–coverage and selective prediction @ 0.8 coverage
    covs, risks = risk_coverage_curve(probs, labels)
    sel_acc = selective_accuracy(probs, labels, target_coverage=0.8)

    out = {
        "ece": float(ece),
        "brier": float(brier),
        "energy_mean": float(energy.mean()),
        "maha_mean": float(maha.mean()),
        "selective_acc@0.8": float(sel_acc),
        "risk_coverage": {
            "coverage": covs.tolist(),
            "risk": risks.tolist()
        }
    }
    path = Path('reports/production_hardening_demo.json')
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
