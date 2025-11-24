# TLS + Policy Utility Integration (Quickstart)

```python
from tfan.train.hooks import build_landmark_masks
from tfan.policy.utility_head import PreferenceUtilityHead
from tfan.fdt.controller import HomeostaticController

# token_embeddings: [B,N,d] from encoder
masks = build_landmark_masks(token_embeddings, keep_ratio=0.30, 
                             mode="rp", per_head=True, n_heads=8, seed=0)  # [B,8,N]
# pass `masks` into your SSA attention to restrict keys per head

# policy preferences
utility = PreferenceUtilityHead(ctx_dim=64, m_objectives=4)
prefs, logits = utility(ctx_vec, temperature=T_eff)  # simplex weights
# combine objectives e.g., L = sum_i prefs[i] * L_i

# FDT-guarded temperature/learning rate
ctrl = HomeostaticController(cfg)
lr_new, T_new = ctrl.step(fdr_ratio, lr_curr, temp_curr)
```
