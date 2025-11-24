from tfan.telemetry import latency_percentiles
from tfan.topo import wasserstein_gap_pct

def test_topology_gap():
    assert wasserstein_gap_pct(0.98, 1.0) <= 2.0

def test_latency_math():
    P = latency_percentiles([1,2,3,4,5,100])
    assert P["p95"] >= P["p50"]
