import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class PADThresholds:
    p: float = 0.0
    a: float = 0.0
    d: float = 0.0

class PADGate:
    def __init__(self, thresholds: PADThresholds):
        self.thr = thresholds

    def open(self, pad: Tuple[float, float, float], signal: float) -> bool:
        p,a,d = pad
        return (p >= self.thr.p) and (a >= self.thr.a) and (d >= self.thr.d) and (signal > 0)
