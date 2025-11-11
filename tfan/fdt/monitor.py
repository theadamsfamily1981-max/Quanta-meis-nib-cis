from collections import deque

def _mean(values):
  if not values:
    return 0.0
  return sum(values) / len(values)

def _variance(values):
  if not values:
    return 0.0
  mean = _mean(values)
  return sum((v - mean) ** 2 for v in values) / len(values)

class FluctuationDissipationMonitor:
  def __init__(self, window=50):
    self.grad_vars = deque(maxlen=window)
    self.lr_hist = deque(maxlen=window)

  def update(self, batch_grads_flat, lr):
    grads = list(batch_grads_flat)
    self.grad_vars.append(_variance(grads))
    self.lr_hist.append(lr)

  @property
  def fluctuation(self):
    return float(_mean(self.grad_vars))

  @property
  def dissipation(self):
    mean_lr = _mean(self.lr_hist)
    return float(mean_lr if mean_lr else 1e-8)

  @property
  def fdr_ratio(self):
    return self.fluctuation / max(self.dissipation, 1e-8)
