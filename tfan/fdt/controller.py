class HomeostaticController:
  def __init__(self, cfg):
    self.cfg = cfg
    self._lr_base = None

  def set_base_lr(self, lr):
    self._lr_base = lr

  def step(self, fdr_ratio, lr_curr, temp_curr):
    if fdr_ratio > self.cfg.fdt['fdr_hot']:
      lr_new = max(1e-6, lr_curr * self.cfg.fdt['lr_mult_hot'])
    elif fdr_ratio < self.cfg.fdt['fdr_cold']:
      lr_new = lr_curr * self.cfg.fdt['lr_mult_cold']
    else:
      lr_new = lr_curr

    temp_target = temp_curr * (1.0 + 0.25 * (fdr_ratio - 1.0))
    temp_min = self.cfg.fdt['temp_min']
    temp_max = self.cfg.fdt['temp_max']
    if temp_target < temp_min:
      temp_target = temp_min
    elif temp_target > temp_max:
      temp_target = temp_max

    return float(lr_new), float(temp_target)
