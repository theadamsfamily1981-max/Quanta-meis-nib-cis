"""Tests for attention module."""
import pytest
import torch

from tfan.attention import RadialSparseMask, SparseMultiHeadAttention


def test_radial_sparse_mask_creation():
    """Test radial sparse mask generator initialization."""
    mask_gen = RadialSparseMask(num_heads=8, keep_ratio=0.33, radius_scale=2.0)

    assert mask_gen.num_heads == 8
    assert mask_gen.keep_ratio == 0.33
    assert mask_gen.radius_scale == 2.0


def test_sparse_attention_forward():
    """Test sparse attention forward pass."""
    B, T, D = 2, 100, 64
    num_heads = 4

    attn = SparseMultiHeadAttention(
        embed_dim=D,
        num_heads=num_heads,
        keep_ratio=0.33,
        use_flash=False  # Use native for testing
    )

    x = torch.randn(B, T, D)

    with torch.no_grad():
        out = attn(x)

    assert out.shape == (B, T, D)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_sparse_attention_different_keep_ratios():
    """Test attention with different keep ratios."""
    B, T, D = 1, 50, 32
    num_heads = 2

    for keep_ratio in [0.2, 0.33, 0.5]:
        attn = SparseMultiHeadAttention(
            embed_dim=D,
            num_heads=num_heads,
            keep_ratio=keep_ratio,
            use_flash=False
        )

        x = torch.randn(B, T, D)

        with torch.no_grad():
            out = attn(x)

        assert out.shape == (B, T, D)
        assert not torch.isnan(out).any()


def test_landmark_selection():
    """Test landmark selection per head."""
    B, H, T, D_head = 2, 4, 100, 16

    attn = SparseMultiHeadAttention(
        embed_dim=H * D_head,
        num_heads=H,
        keep_ratio=0.33,
        use_flash=False
    )

    K = torch.randn(B, H, T, D_head)

    landmarks = attn.select_landmarks_per_head(K)

    assert landmarks.shape == (B, H, int(T * 0.33))
    assert landmarks.dtype == torch.long
    assert (landmarks >= 0).all()
    assert (landmarks < T).all()


def test_sparse_attention_gradient_flow():
    """Test that gradients flow through sparse attention."""
    B, T, D = 1, 50, 32
    num_heads = 2

    attn = SparseMultiHeadAttention(
        embed_dim=D,
        num_heads=num_heads,
        keep_ratio=0.33,
        use_flash=False
    )

    x = torch.randn(B, T, D, requires_grad=True)

    out = attn(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_mask_application():
    """Test mask application to attention scores."""
    mask_gen = RadialSparseMask(num_heads=2, keep_ratio=0.33)

    B, H, T_q, T_k = 1, 2, 10, 10
    attn_scores = torch.randn(B, H, T_q, T_k)
    mask = torch.rand(B, H, T_q, T_k) > 0.5  # Random boolean mask

    masked_scores = mask_gen.apply_mask(attn_scores, mask)

    # Check that masked positions are -inf
    assert torch.isinf(masked_scores[~mask]).all()
    assert (masked_scores[~mask] < 0).all()

    # Check that unmasked positions are unchanged
    assert torch.allclose(masked_scores[mask], attn_scores[mask])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sparse_attention_cuda():
    """Test sparse attention on CUDA."""
    B, T, D = 1, 100, 64
    num_heads = 4

    attn = SparseMultiHeadAttention(
        embed_dim=D,
        num_heads=num_heads,
        keep_ratio=0.33,
        use_flash=False
    ).cuda()

    x = torch.randn(B, T, D).cuda()

    with torch.no_grad():
        out = attn(x)

    assert out.device.type == "cuda"
    assert out.shape == (B, T, D)
    assert not torch.isnan(out).any()
