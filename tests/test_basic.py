from common.pad import PAD

import math
import random

# 20 lightweight tests — stubs focusing on behavior, not specific frameworks


def test_pad_clamp():
    p = PAD(2.0, -5.0, 0.5).clamp()
    assert p.pleasure == 1.0 and p.arousal == -1.0 and p.dominance == 0.5


def test_pad_gate_opens_when_all_above():
    p = PAD(0.2, 0.3, 0.4)
    assert p.gate(1.0, (0.1, 0.1, 0.1)) is True


def test_pad_gate_closes_on_signal():
    p = PAD(0.9, 0.9, 0.9)
    assert p.gate(0.0, (0.1, 0.1, 0.1)) is False


from common.rank import rank_schedule


def test_rank_schedule_monotone_decay():
    seq = rank_schedule(64, 9)
    assert seq[0] >= seq[4] >= seq[8]


from quanta.router import router_temperature_sweep


def test_router_tau_positive():
    import pytest

    with pytest.raises(ValueError):
        router_temperature_sweep(0)


from meis.leak_probe import LeakProbe


def test_leak_probe_attenuation_bounds():
    lp = LeakProbe()
    assert 0.0 <= lp.attenuation(0.0) <= 1.0
    assert 0.0 <= lp.attenuation(1.0) <= 1.0


# Create 15 more ultra-light tests to reach 20 total


def test_pad_threshold_edge():
    p = PAD(0.1, 0.1, 0.1)
    assert p.gate(1.0, (0.1, 0.1, 0.1))


def test_pad_negative_thresholds():
    p = PAD(-0.2, -0.3, -0.4)
    assert p.gate(1.0, (-0.5, -0.5, -0.5))


def test_rank_schedule_never_zero():
    seq = rank_schedule(8, 10)
    assert all(x >= 1 for x in seq)


def test_router_entropy_proxy_range():
    v = router_temperature_sweep(0.5)
    assert 0.0 <= v <= 10.0


def test_leak_probe_escalation_linear():
    lp = LeakProbe()
    assert lp.escalate(0) == 1.0 and lp.escalate(2) > lp.escalate(1)


def test_pad_gate_requires_all_three():
    p = PAD(0.2, 0.2, 0.0)
    assert not p.gate(1.0, (0.1, 0.1, 0.0))


def test_pad_clamp_idempotent():
    p = PAD(2, 2, 2).clamp().clamp()
    assert (p.pleasure, p.arousal, p.dominance) == (1.0, 1.0, 1.0)


def test_rank_schedule_thirds():
    seq = rank_schedule(9, 6)
    assert seq[:2] == [9, 9] and seq[2:4] == [4, 4] and seq[4:] == [1, 1]


def test_router_tau_inverse_relation():
    a = router_temperature_sweep(0.5)
    b = router_temperature_sweep(1.0)
    assert a > b


def test_leak_probe_risk_monotonic():
    lp = LeakProbe()
    assert lp.attenuation(0.2) > lp.attenuation(0.8)


def test_pad_randomized_gate_stats():
    random.seed(0)
    opened = 0
    for _ in range(100):
        p = PAD(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
        if p.gate(1.0, (0.0, 0.0, 0.0)):
            opened += 1
    assert 0 < opened < 100


def test_rank_schedule_length():
    seq = rank_schedule(16, 7)
    assert len(seq) == 7


def test_router_tau_sensible():
    assert router_temperature_sweep(2.0) < router_temperature_sweep(1.0)


def test_leak_probe_types():
    lp = LeakProbe()
    assert isinstance(lp.attenuation(0.3), float) and isinstance(lp.escalate(2), float)


def test_pad_gate_off_when_any_low():
    p = PAD(0.9, 0.9, -0.1)
    assert not p.gate(1.0, (0.0, 0.0, 0.0))
