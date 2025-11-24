import torch
from tfan.policy.utility_head import PreferenceUtilityHead


def test_simplex_output():
  head = PreferenceUtilityHead(ctx_dim=16, m_objectives=4)
  ctx = torch.randn(3, 16)
  p, _ = head(ctx, temperature=0.9)
  assert p.shape == (3, 4)
  s = torch.sum(p, dim=-1)
  assert torch.allclose(s, torch.ones_like(s), atol=1e-5)
  assert torch.all(p >= 0)
