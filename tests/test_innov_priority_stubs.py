import json
from experiments.innov import run_innov_001, run_innov_016, run_innov_036


def test_innov_priority_runs():
    res1 = run_innov_001()
    res2 = run_innov_016()
    res3 = run_innov_036()
    # minimal schema assertions
    for r in (res1, res2, res3):
        assert "suite_id" in r and isinstance(r["suite_id"], str)
    # ensure JSON-serializable
    json.dumps({"r1": res1, "r2": res2, "r3": res3})
