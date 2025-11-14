import os, json, time, torch
from .tls import hybrid_landmarks
from .ssa import ssa_forward
from .fdt import PID, EPRCV
from .pgu import PGUCache
from .metrics import fit_power, percentile
from .io import new_artifact, write_json

def run_smoke(device="cuda:0" if torch.cuda.is_available() else "cpu"):
    torch.manual_seed(13)
    dev = torch.device(device)

    # 1) synthetic tokens
    T = 8192
    D = 512
    X = torch.randn(T, D, device=dev) * 0.2
    k = int(0.33*T)

    # 2) TLS -> keep indices
    keep = hybrid_landmarks(X, k=k, persistence_weight=0.7)

    # 3) SSA-sim forward
    Q = torch.randn(T, D, device=dev)
    K = X
    V = torch.randn(T, D, device=dev)
    out = ssa_forward(Q, K, V, keep, scale=(1.0/(D**0.5)))
    _ = out.norm()

    # 4) FDT: drive EPR CV to spec with PID effect on 'temperature'
    pid = PID(kp=0.25, ki=0.03, kd=0.08, clamp=(0.5, 2.0))
    cvmon = EPRCV(window=120)
    T_eff = 1.2
    eprs = []
    for _ in range(180):
        # synthetic epr that responds to T_eff (lower T_eff -> lower epr)
        epr = max(1e-3, 0.2*T_eff + 0.01*torch.randn((), device=dev).item())
        eprs.append(epr)
        cvmon.push(epr)
        err = (cvmon.cv() - 0.15)
        u = pid.step(-err)                     # reduce T if CV>0.15
        T_eff = 0.9*T_eff + 0.1*u              # EMA towards control

    epr_cv = cvmon.cv()

    # 5) PGU latency (tiny set)
    pgu = PGUCache(timeout_ms=120)
    cases = [
        "(declare-const a Int) (declare-const b Int) (assert (> (+ a b) 5)) (check-sat)",
        "(declare-const x Int) (declare-const y Int) (assert (> (+ x y) 5)) (check-sat)",  # alpha-rename hit
        "(declare-const p Bool) (assert p) (check-sat)"
    ]
    lats = []
    for s in cases:
        r = pgu.check(s)
        lats.append(r["ms"])
    p95 = percentile(lats, 95)

    # 6) memory alpha quick sim (three points)
    Ts = [2000, 6000, 12000]
    alloc = []
    for t in Ts:
        q_tensor = torch.empty((t, D), dtype=torch.float16, device=dev)
        k_tensor = torch.empty((int(0.33*t), D), dtype=torch.float16, device=dev)
        v_tensor = torch.empty_like(k_tensor)
        torch.cuda.synchronize() if dev.type == "cuda" else None
        bytes_est = (q_tensor.numel()+k_tensor.numel()+v_tensor.numel())*2
        alloc.append(bytes_est)
        del q_tensor,k_tensor,v_tensor
    a, alpha = fit_power(Ts, alloc)

    report = {
        "device": str(dev),
        "num_tokens": T, "k": k, "keep_ratio": 0.33,
        "epr_cv": epr_cv,
        "pgu_p95_ms": p95,
        "alpha": alpha
    }
    out = new_artifact("smoke_report")
    write_json(out, report)
    ok = (alpha < 1.0) and (p95 <= 200.0) and (epr_cv <= 0.15)
    return ok, str(out), report
