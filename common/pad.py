from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

@dataclass
class PAD:
    pleasure: float
    arousal: float
    dominance: float

    def clamp(self) -> "PAD":
        self.pleasure = max(-1.0, min(1.0, self.pleasure))
        self.arousal = max(-1.0, min(1.0, self.arousal))
        self.dominance = max(-1.0, min(1.0, self.dominance))
        return self

    def gate(self, signal: float, threshold: Tuple[float, float, float]) -> bool:
        """Simple PAD gating: open if all three exceed their thresholds.
        threshold: (p_thr, a_thr, d_thr) in [-1,1]
        """
        p_thr, a_thr, d_thr = threshold
        p_ok = self.pleasure >= p_thr
        a_ok = self.arousal >= a_thr
        d_ok = self.dominance >= d_thr
        return bool(p_ok and a_ok and d_ok and signal > 0.0)
