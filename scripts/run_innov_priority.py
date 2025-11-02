#!/usr/bin/env python3
from __future__ import annotations
import json
from experiments.innov import run_innov_001, run_innov_016, run_innov_036


if __name__ == "__main__":
    results = {
        "INNOV-001": run_innov_001(),
        "INNOV-016": run_innov_016(),
        "INNOV-036": run_innov_036(),
    }
    print(json.dumps(results, indent=2))
