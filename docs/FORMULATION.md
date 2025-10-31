# QUANTA Meta-Cognition — Formal Specification (Concise)

This document formalizes gates, objectives, and runtime policy for QUANTA Meta-Cognition. It matches the experiment data and CI gates in this repo.

## Gates (targets)
- Accuracy (ID): **≥ 0.75** (dev), **≥ 0.78** (prod guard band)
- ECE: **≤ 0.03**
- OOD separation (energy): **≥ 0.05** (dev), **≥ 0.06** (prod)
- Selective accuracy @ 80% coverage: **≥ 0.80**
- Latency p50: **≤ 50 ms**
- Forgetting (abs): **≤ 0.10**

## Key definitions
- Logits: f_θ(x) ∈ R^K
- Temp-scaled softmax: p_T(y|x) = softmax(f_θ(x)/T)
- Confidence: s_T(x) = max_y p_T(y|x)
- Energy (OOD): E_T(x) = − T * log ∑_k exp(f_θ(x)_k / T)
- OOD separation: Δ = E[ E_T(x) | OOD ] − E[ E_T(x) | ID ]

## Runtime decision policy
1) Compute calibration (T), confidence s_T, and energy E_T.
2) If E_T ≥ γ → **abstain/route** (OOD).
3) Else if s_T < τ_sel (for 80% coverage) → **abstain**.
4) Else return argmax p_T.

## Router temperature (PAD gate)
τ_route ← clamp( τ0 * (1 + 0.25·max(0, A)) * (1 − 0.20·max(0, −P)), 0.5, 2.0 )

## CI gate-check (illustrative)
```python
# After running run_experiments.py which writes experiment_summary.json
from math import fabs
import json
with open('experiment_summary.json') as f: m = json.load(f)
a = m['after']; g = m['gates']
assert a['accuracy'] >= 0.75
assert a['ece'] <= 0.03
assert a['ood_sep'] >= 0.05
assert a['selective_acc@80'] >= 0.80
assert a['latency_ms'] <= 50
assert abs(m['after'].get('forgetting', 0.0)) <= 0.10
```

See `configs/gates.yaml` and `scripts/check_gates.py` for the real implementation used by CI.
