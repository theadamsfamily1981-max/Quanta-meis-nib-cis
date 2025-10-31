import math

class LeakProbe:
    def __init__(self, target_auc_max: float = 0.55):
        self.target = target_auc_max

    def attenuation(self, risk: float) -> float:
        """Simple attenuation: more risk -> more attenuation (0..1).
        """
        r = min(1.0, max(0.0, risk))
        return 1.0 - r * 0.8

    def escalate(self, level: int) -> float:
        return 1.0 + 0.05 * max(0, level)
