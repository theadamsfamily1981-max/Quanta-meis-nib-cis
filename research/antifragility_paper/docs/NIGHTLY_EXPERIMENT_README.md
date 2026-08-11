# 🌙 Nightly Experiment Runner

This GitHub Action executes all antifragility experiment phases every night
at **05:00 UTC**, merges results into
`research/antifragility_paper/data/merged_experiment_summary.csv`, and then
triggers the paper build workflow.

### Included Scripts
- `phase3_ph_stability.py`
- `phase4_architecture_morph.py`
- `phase5_domain_shift.py`

### Outputs
- Updated CSV summary of all experiment metrics.
- Automatically rebuilt paper PDF with fresh figures/tables.

You can manually trigger this workflow under **Actions → Nightly Experiment Runner → Run workflow**.
