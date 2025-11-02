#!/usr/bin/env python3
"""
Bench wrapper: ensures reports/ exists and writes innov_priority.json for CI artifact.
"""
import json, pathlib, sys
from experiments.innov import run_innov_001, run_innov_016, run_innov_036

ROOT = pathlib.Path(__file__).parents[1]
REPORTS = ROOT / 'reports'
REPORTS.mkdir(exist_ok=True)

res = {
    'INNOV-001': run_innov_001(),
    'INNOV-016': run_innov_016(),
    'INNOV-036': run_innov_036(),
}

out = REPORTS / 'innov_priority.json'
out.write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
