def wasserstein_gap_pct(proxy_score: float, exact_score: float) -> float:
    if exact_score == 0: return 0.0
    return abs(proxy_score - exact_score) / abs(exact_score) * 100.0
