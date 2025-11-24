from pathlib import Path
import json
import subprocess
import sys

RUNNER = Path("experiments/run_experiments.py").resolve()
CFG = Path("experiments/configs/example_mam.yaml").resolve()


def test_runner_smoke(tmp_path):
    out = tmp_path / "summary.json"
    cmd = [sys.executable, str(RUNNER), "--config", str(CFG), "--out", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out.exists(), proc.stdout + proc.stderr
    data = json.loads(out.read_text())
    assert data.get("count", 0) >= 1
    assert data.get("ok") is True
    assert "avg" in data
