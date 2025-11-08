# Repository Summary

## TFAN Modules
- `tfan_phase_I.summarize_inputs` averages initial probe data and returns zero for empty probes.
- `tfan_phase_II.normalize_series` scales data to a unit range, enabling fair comparisons across probes.
- `tfan_phase_II.phase_two_projection` feeds normalized data through Phase I to maintain continuity between phases.

## GRTES Framework
- `grtes_framework.phase_i.compute_baseline` generates a baseline value from incoming signals.
- `grtes_framework.phase_ii.compute_adjusted_baseline` boosts the baseline for proactive adjustments.

## Automation
- `.github/workflows/ci.yml` ensures Python modules remain syntactically valid via `python -m compileall`.
