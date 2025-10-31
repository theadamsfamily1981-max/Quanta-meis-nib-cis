import numpy as np

# Energy score for OOD (lower is more confident in-distribution)
# E(x) = -logsumexp(logits)


def energy_score(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=1, keepdims=True)
    return -np.log(np.exp(logits - m).sum(axis=1)) - m.squeeze(1)


# Class-conditional Mahalanobis distance given class means and shared covariance

def mahalanobis_scores(features: np.ndarray, means: np.ndarray, cov: np.ndarray) -> np.ndarray:
    inv = np.linalg.pinv(cov)
    scores = []
    for mu in means:
        diff = features - mu
        # quadratic form per sample
        q = np.einsum('bi,ij,bj->b', diff, inv, diff)
        scores.append(q)
    return np.stack(scores, axis=1)  # (B, C)


# Risk-coverage: return pairs (coverage, risk=1-acc) over thresholds on confidence

def risk_coverage_curve(probs: np.ndarray, labels: np.ndarray, n_points: int = 21):
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.float32)
    qs = np.linspace(0.0, 1.0, n_points)
    covs, risks = [], []
    for q in qs:
        thr = np.quantile(conf, 1 - q)  # keep top-q confidence
        keep = conf >= thr
        coverage = float(keep.mean())
        risk = float(1.0 - correct[keep].mean()) if coverage > 0 else 0.0
        covs.append(coverage)
        risks.append(risk)
    return np.array(covs), np.array(risks)


# Selective accuracy at desired coverage: choose confidence threshold s.t. coverage≈target

def selective_accuracy(probs: np.ndarray, labels: np.ndarray, target_coverage: float) -> float:
    conf = probs.max(axis=1)
    thr = np.quantile(conf, 1 - target_coverage)
    keep = conf >= thr
    if keep.sum() == 0:
        return 0.0
    return float((probs[keep].argmax(axis=1) == labels[keep]).mean())
