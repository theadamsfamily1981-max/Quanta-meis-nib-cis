def enable_hyperbolic(pred_ndcg_gain: float, overhead: float, cfg) -> bool:
  return (pred_ndcg_gain >= cfg.ctd['ndcg_gain_min'] 
          and overhead <= cfg.ctd['riemann_overhead_max'])
