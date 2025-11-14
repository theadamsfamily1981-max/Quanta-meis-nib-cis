import torch

def ssa_forward(Q, K, V, keep_idx, scale=1.0):
    """
    Very lightweight SSA 'sim':
    - Restrict K,V to landmarks (keep_idx).
    - Attend each query against those landmarks only.
    Q: [Tq, D], K: [Tk, D], V: [Tk, Dv], keep_idx: [k]
    Returns: [Tq, Dv]
    """
    Kk = K.index_select(0, keep_idx)
    Vk = V.index_select(0, keep_idx)
    att = (Q @ Kk.T) * scale
    att = torch.softmax(att, dim=-1)
    return att @ Vk
