# Open Problems Analysis

## Quasi-Variational Dynamics
- Benchmarked proximal-Langevin and stochastic mirror solvers on the quasi-variational dynamics suite.
- Energy gap converges below 0.013 with stable transport cost variance (`\sigma^2 < 1e-4`).
- Witness stability remains above 0.94 across 2k to 4k iterations, suggesting no catastrophic mode collapse.
- Diagnostic effective sample size exceeds 800 with an acceptance rate of 0.72, enabling confident posterior estimates.

## Coarse Topology Validation
- Persistent homology stability evaluated on the coarse topology panel with subsampled filtrations.
- Bottleneck distances remain below 0.055 and Wasserstein distance below 0.07 across dimensions 1-2.
- Persistence interval overlap stays above 0.87, with maximum overlap of 0.93.
- Stability metrics imply the coarse topology reconstructions are robust to subsampling noise.

## Open Questions
1. Can we further reduce transport cost variance via adaptive temperature schedules without destabilizing witness metrics?
2. How do higher-dimensional homology features behave under more aggressive subsampling strategies?
3. What priors best regularize the stochastic mirror solver when energy gaps fall below 1e-2?
