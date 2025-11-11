"""Validation gates mirroring the expectations in VALIDATION.md."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_landmark_eval import compute_landmark_speedups
from scripts.run_lora_efficiency_eval import compute_lora_efficiency
from scripts.run_seq_learning_eval import simulate_seq_learning
from scripts.run_unified_embedding_eval import compute_unified_embedding_metrics


@pytest.fixture(scope="module")
def validation_doc() -> str:
    return Path("VALIDATION.md").read_text(encoding="utf-8")


def test_unified_embedding_cosine_gate(validation_doc: str) -> None:
    assert "cosine" in validation_doc.lower(), "VALIDATION.md must mention cosine threshold"
    metrics = compute_unified_embedding_metrics()
    for key, value in metrics.items():
        assert value >= 0.95, f"{key} cosine {value:.3f} below 0.95 gate"


def test_surprise_replay_forgetting_gate(validation_doc: str) -> None:
    assert "forgetting" in validation_doc.lower(), "VALIDATION.md must document forgetting gate"
    results = simulate_seq_learning(["task_a", "task_b", "task_c"])
    avg_forgetting = sum(r.forgetting for r in results) / len(results)
    assert avg_forgetting <= 0.005, f"Average forgetting {avg_forgetting:.4f} exceeds 0.5% gate"


def test_lora_efficiency_gate(validation_doc: str) -> None:
    assert "lora" in validation_doc.lower(), "VALIDATION.md must document LoRA gate"
    metrics = compute_lora_efficiency({"vision": 2048, "audio": 1024, "text": 3072})
    assert metrics["reduction_fraction"] >= 0.95, "LoRA reduction must be at least 95%"


def test_landmark_speedup_gate(validation_doc: str) -> None:
    assert "speedup" in validation_doc.lower(), "VALIDATION.md must document speedup gate"
    metrics = compute_landmark_speedups([128, 256, 512, 1024, 2048])
    length_to_speed = dict(zip(metrics["sequence_lengths"], metrics["speedup"]))
    assert length_to_speed[2048] >= 16.0, "Speedup at seq length 2048 must be ≥ 16×"
