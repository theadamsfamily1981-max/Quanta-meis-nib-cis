def enable_hyperbolic(predicted_ndcg_gain: float, riemann_overhead: float, cfg) -> bool:
    return (predicted_ndcg_gain >= cfg.ctd_enable_ndcg_gain_min and
            riemann_overhead <= cfg.ctd_riemann_overhead_max)
