import torch
from tfan.tls.selector import select_landmarks


def build_landmark_masks(token_embeddings: torch.Tensor, keep_ratio: float, mode: str, per_head: bool, n_heads: int, seed: int = 0):
  """
  token_embeddings: [B, N, d] (torch)
  returns: bool mask [B, H, N] with H=1 or n_heads
  """
  B, N, d = token_embeddings.shape
  H = n_heads if per_head else 1
  masks = []
  for b in range(B):
    X = token_embeddings[b].detach().cpu().numpy()
    k = max(1, int(keep_ratio * N))
    m, _scores = select_landmarks(X, k, mode=mode, per_head=per_head, n_heads=n_heads, seed=seed + b)
    # m: [H, N] np.bool_
    masks.append(torch.from_numpy(m).to(token_embeddings.device))
  return torch.stack(masks, dim=0)  # [B, H, N]
