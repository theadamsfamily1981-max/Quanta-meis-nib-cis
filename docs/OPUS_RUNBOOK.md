# Opus Runbook

This repo is structured for AI-assisted iteration (Opus/Claude/GPT).

## PR Rules
- ≤ **1,000 LOC** per PR.
- Add/update **`DESIGN.md`** and **`ACCEPTANCE.md`** for any new component.
- Include **golden tests** and update **JSON artifacts** where relevant.
- CI must pass: ruff, black, mypy (lenient), pytest.

## Commands
- Bench (small slices): _manual for now_ via `scripts/run_production_suite.py`.
- Metrics artifacts: see `reports/production_hardening_demo.json`.

## Layout
- `core/` math, adapters, losses.
- `quanta/` routing, temperature, resource-aware logic.
- `meis/` tagging + consolidation (STC, SVD).
- `cis/` PAD gating, meta-monitoring.
- `datasets/` bridges (MELD/IEMOCAP) + synthetic.
- `experiments/` harness, ood, metrics, configs.
- `serving/` FastAPI (stub).
- `ui/` dashboard (placeholder).
- `tools/` env preflight, gpu selector, smoke run.
- `tests/` unit + e2e + goldens.

## Accept/Reject Checklist
- [ ] Clear spec in `docs/`
- [ ] Golden tests present
- [ ] CI green
- [ ] Size budget respected
