.ONESHELL:
PY=python

init:
	python -m venv .venv && . .venv/bin/activate && pip -q install -U pip && pip -q install -r requirements.txt

smoke-3090:
	$(PY) -m tfan.runner --config configs/prod_3090.yaml --smoke

smoke-3060:
	$(PY) -m tfan.runner --config configs/prod_3060.yaml --smoke

bench-alpha:
	$(PY) scripts/bench_memory_alpha.py --lengths 1000 2000 5000 10000 20000 \
		--keep-ratio 0.33 --resident-pages 4 --page-size 2048 --out artifacts/bench_memory_alpha.json

pgu-latency:
	$(PY) scripts/latency_pgu_replay.py --path tests/pgu_cases --timeout-ms 120 --trials 5

epr-audit:
	$(PY) scripts/epr_cv_monitor.py --file logs/epr.csv --col epr --window 120 --threshold 0.15 --grace 60
