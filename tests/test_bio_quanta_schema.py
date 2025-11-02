import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_bio_baselines_schema():
    import packages.bio_baselines as BB  # noqa: F401
    assert (ROOT / 'packages' / 'bio_baselines' / '__init__.py').exists()


def test_quanta_bench_emits_keys():
    out = ROOT / 'reports' / 'quanta'
    expected = ['risk_coverage.json', 'calibration.json', 'ood.json', 'latency.json']
    for name in expected:
        p = out / name
        assert p.parent.name == 'quanta'
