"""
TF-A-N Pareto Auto-Runner with EHVI

Multi-objective optimization with Expected Hypervolume Improvement:
- Batch run configurations (grid search or Bayesian optimization)
- Compute Pareto frontier
- EHVI-based selection for next candidates
- Interactive dashboard for visualization

Hard gates:
- ≥6 non-dominated points with crowding ε=0.05
- End-to-end wall-time ≤6h on 1×RTX 3090

Objectives (example):
- Minimize: latency, memory, cost
- Maximize: accuracy, throughput
"""

from .runner import ParetoRunner, RunConfig
from .ehvi import compute_ehvi, pareto_frontier
from .summarize import summarize_results, export_dashboard

__all__ = [
    'ParetoRunner',
    'RunConfig',
    'compute_ehvi',
    'pareto_frontier',
    'summarize_results',
    'export_dashboard'
]
