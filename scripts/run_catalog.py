#!/usr/bin/env python3
import json, sys, pathlib

if __name__ == "__main__":
    root = pathlib.Path(__file__).parents[1]
    manifest = json.loads((root / "experiments" / "catalog" / "nib_suite_45.json").read_text())
    print(json.dumps({"count": len(manifest["experiments"]), "ids": [e["id"] for e in manifest["experiments"]]}, indent=2))
