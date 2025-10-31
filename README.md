# QUANTA / MEIS / NIB / CIS

**Goal:** Continual-learning architecture with PAD-gated adapters, rank scheduling, and NIB governance.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
pytest -q
```

## Structure
- `quanta/` partitioning + adapters
- `meis/` meta-governor & NIB loop
- `cis/` emotion & context inference
- `tests/` unit/integration; add fixtures for synthetic tasks

## Experiments
- `scripts/run_experiments.py`
