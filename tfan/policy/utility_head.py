import torch
import torch.nn as nn
import torch.nn.functional as F


class PreferenceUtilityHead(nn.Module):
  """
  Maps a context vector -> preference weights on M objectives (simplex).
  Temperature T is guarded externally by FDT signals.
  """
  def __init__(self, ctx_dim: int, m_objectives: int, hidden: int = 256):
    super().__init__()
    self.m = m_objectives
    self.net = nn.Sequential(
      nn.Linear(ctx_dim, hidden),
      nn.GELU(),
      nn.Linear(hidden, hidden),
      nn.GELU(),
      nn.Linear(hidden, m_objectives),
    )

  def forward(self, ctx, temperature: float = 1.0):
    logits = self.net(ctx) / max(1e-6, temperature)
    prefs = F.softmax(logits, dim=-1)  # simplex on M objectives
    return prefs, logits
