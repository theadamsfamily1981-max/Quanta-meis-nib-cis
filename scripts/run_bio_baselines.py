#!/usr/bin/env python3
from __future__ import annotations
import json
from packages.bio_baselines import run_all

if __name__ == '__main__':
    print(json.dumps(run_all(), indent=2))
