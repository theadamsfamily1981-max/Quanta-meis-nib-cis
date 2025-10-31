# Integration Guide

This repo now includes:

- `docs/FORMULATION.md` — formal spec and gate math
- `configs/gates.yaml` — dev/prod thresholds (guard bands)
- `experiments/default.yaml` — defaults (LR=1e-3, T=1.5, batch=32)
- `run_experiments.py` — generates `experiment_summary.json`
- Minimal CI workflow `gate-check.yml` scaffold

## Local run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r env/requirements.txt pyyaml
python run_experiments.py
cat experiment_summary.json
```

## Next steps
- Expand `.github/workflows/gate-check.yml` to run gate checks and upload artifacts.
- Optionally add a strict (prod) step that enforces guard bands (acc ≥ 0.78, OOD ≥ 0.06).
