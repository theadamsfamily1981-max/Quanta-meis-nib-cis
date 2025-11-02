:format
	black .

:lint
	ruff check .

:test
	pytest -q

:rc
	python scripts/run_production_suite.py || true

:catalog
	python scripts/run_catalog.py
