from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

from tfan import alpha_probe_loss, jt_fan_loss, landmark_attention_loss, run_pipeline


def test_losses_are_computable() -> None:
    rng = random.Random(123)
    embeddings = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(4)]
    activations = [rng.gauss(0, 1) for _ in range(3)]
    targets = [rng.gauss(0, 1) for _ in range(3)]
    energy = [rng.gauss(0, 1) for _ in range(3)]

    assert landmark_attention_loss(embeddings) >= 0
    assert 0 <= alpha_probe_loss(activations, targets) <= 2
    assert jt_fan_loss(energy) >= 0


def test_pipeline_saves_plots(tmp_path: Path) -> None:
    result = run_pipeline(save_plots=tmp_path, k_values=[1, 2, 3])
    assert result.jt_fan >= 0

    expected = {
        "pareto_acc_vs_diss_default.png",
        "throughput_vs_k_default.png",
        "nfl_structured_vs_scrambled.png",
    }
    produced = {p.name for p in tmp_path.iterdir()}
    assert expected.issubset(produced)


def test_cli_entrypoint(tmp_path: Path) -> None:
    output_dir = tmp_path / "plots"
    cmd = [sys.executable, "-m", "tfan.cli", "--save-plots", str(output_dir)]
    subprocess.check_call(cmd)

    assert (output_dir / "pareto_acc_vs_diss_default.png").exists()
    assert (output_dir / "throughput_vs_k_default.png").exists()
    assert (output_dir / "nfl_structured_vs_scrambled.png").exists()
