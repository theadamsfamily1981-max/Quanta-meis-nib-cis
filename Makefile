# Simple dev automation for QUANTA / MEIS / NIB

PY ?= python
PIP ?= pip
VENV ?= .venv
ACT := . $(VENV)/bin/activate

.PHONY: help venv install dev test lint fmt experiments clean ci

help:
@echo "Targets: venv install dev test lint fmt experiments clean ci"

venv:
$(PY) -m venv $(VENV) || true
$(ACT); $(PY) -m pip install -U pip

install: venv
$(ACT); $(PIP) install -r requirements.txt

dev: install
$(ACT); $(PIP) install -r dev-requirements.txt || true
$(ACT); which pytest || true
$(ACT); which ruff || true

# Run unit tests
test:
$(ACT); pytest -q

# Lint and format (Ruff)
lint:
$(ACT); ruff check .

fmt:
$(ACT); ruff check --fix .

# Run all experiment configs and write artifacts/summary.json
experiments:
$(ACT); $(PY) experiments/run_experiments.py --all --out artifacts/summary.json
@echo "Artifacts written to artifacts/summary.json"

# CI convenience: lint + tests
ci:
$(MAKE) fmt
$(MAKE) test

clean:
rm -rf $(VENV) __pycache__ **/__pycache__ .pytest_cache artifacts *.egg-info .ruff_cache
