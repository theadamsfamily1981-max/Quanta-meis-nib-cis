from dataclasses import dataclass


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


@dataclass
class PAD:
    pleasure: float
    arousal: float
    dominance: float


def route_temperature(pad: PAD, base_tau: float = 1.0) -> float:
    tau = base_tau
    tau *= 1.0 + 0.25 * max(0.0, pad.arousal)
    tau *= 1.0 - 0.20 * max(0.0, -pad.pleasure)
    return clamp(tau, 0.5, 2.0)
