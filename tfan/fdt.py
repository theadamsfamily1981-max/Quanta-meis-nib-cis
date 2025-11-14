import time, statistics

class PID:
    def __init__(self, kp, ki, kd, clamp=(0.1, 10.0)):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.ei = 0.0
        self.prev_e = 0.0
        self.clamp = clamp

    def step(self, error):
        self.ei += error
        ed = error - self.prev_e
        self.prev_e = error
        u = self.kp*error + self.ki*self.ei + self.kd*ed
        lo, hi = self.clamp
        return max(lo, min(hi, u))

class EPRCV:
    def __init__(self, window=120):
        self.window = window
        self.buf = []

    def push(self, v):
        self.buf.append(float(v))
        if len(self.buf) > self.window: self.buf.pop(0)

    def cv(self):
        if len(self.buf) < 5: return 0.0
        mu = statistics.fmean(self.buf)
        if abs(mu) < 1e-12: return 0.0
        return statistics.pstdev(self.buf)/abs(mu)
