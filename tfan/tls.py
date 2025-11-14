import torch

@torch.no_grad()
def hybrid_landmarks(X: torch.Tensor, k: int, persistence_weight=0.7, seed=13):
    """
    X: [N, D] embeddings (e.g., token states). We simulate a 'persistence' score by
    local density inverse (cheaper than full PH here), blended with max-min spread.
    Returns indices of selected k landmarks.
    """
    N = X.shape[0]
    k = max(1, min(k, N))
    gen = torch.Generator(device=X.device).manual_seed(seed)

    # Cheap "persistence-like" proxy: inverse of local L2 density
    # (High score = structurally salient under this proxy)
    # Compute 8-NN average distance as density proxy
    device_type = 'cuda' if X.device.type == 'cuda' else 'cpu'
    with torch.amp.autocast(device_type=device_type, enabled=False):
        d2 = torch.cdist(X.float(), X.float())  # [N, N]
        kn = 8 if N > 8 else max(1, N-1)
        vals, _ = torch.topk(d2, k=kn, dim=1, largest=False)
        density = vals[:, 1:].mean(dim=1) if kn > 1 else vals[:, 0]
        pers_score = density  # higher => sparser neighborhood => more "persistent"

    # Max-min seeding
    first = torch.randint(0, N, (1,), generator=gen, device=X.device)
    sel = [first.item()]
    while len(sel) < k:
        dist_to_sel = d2[:, sel].min(dim=1).values
        # Blend: persistence + distance from selected
        score = persistence_weight * pers_score + (1 - persistence_weight) * dist_to_sel
        score[sel] = -1e9
        nxt = int(torch.argmax(score).item())
        sel.append(nxt)
    return torch.tensor(sel, device=X.device, dtype=torch.long)
