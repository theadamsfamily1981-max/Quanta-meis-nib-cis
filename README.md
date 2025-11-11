# Quanta-meis-nib-cis
Research for quanta meis nib cis code.

## What changed
- UTCF hooks and lambda defaults tuned for better stability in mixed workloads.
- Added DPH proxy or exact switch for adaptive precision handling.
- Introduced CAT/landmark attention along with TTW + DTW sentry safeguards.
- Enabled SuRe with dual-learner coordination to reduce drift.
- Hardened PGU invariants to keep critical state consistent.

## CI expectations
- LoRA compression must reach at least 95%.
- PH proxy gap should remain within 2% while running at least 3× faster.
- CAT throughput targets ≥10× at 2048 tokens and TTW relative error ≤5%.
- Sentry AUROC should stay ≥0.90 with catastrophic forgetting ≤2%.
- UTCF OOD F1 needs a gain of at least 1.5% with ≤10% overhead.
