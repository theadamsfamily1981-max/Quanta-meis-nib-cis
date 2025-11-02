#!/usr/bin/env python3
# Runs your repo harness with Torch backend if available, falls back to NumPy otherwise.
import os, subprocess, sys, json


def run(cmd):
    print(">>", cmd)
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print(p.stdout)
    if p.returncode != 0:
        print(p.stderr)
    return p.returncode


def main():
    # Prefer torch if available
    try:
        import torch  # noqa: F401
        os.environ["QMNC_BACKEND"] = "torch"
    except Exception:
        os.environ["QMNC_BACKEND"] = "numpy"

    # Run experiment runner
    rc = run("python experiments/runner.py")
    # Run production hardening demo
    rc2 = run("python scripts/run_production_suite.py")
    # Print report file if exists
    try:
        with open("reports/production_hardening_demo.json") as f:
            data = json.load(f)
        print("\nproduction_hardening_demo.json:")
        print(json.dumps(data, indent=2))
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
