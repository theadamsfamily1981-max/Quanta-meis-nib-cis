"""Shared multimodal modules for unified representation learning."""

from __future__ import annotations

from typing import Dict, List, MutableMapping, Optional, Tuple

import torch
from torch import Tensor, nn


class SharedProjection(nn.Module):
    """Project heterogeneous modality embeddings into a shared space.

    The module maintains a bank of per-modality linear projections that map
    arbitrary input dimensions to a common ``hidden_dim``. The projections are
    layer-normalised to stabilise optimisation across modalities.
    """

    def __init__(
        self,
        input_dims: MutableMapping[str, int],
        hidden_dim: int,
        bias: bool = True,
        activation: Optional[nn.Module] = nn.GELU(),
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not input_dims:
            raise ValueError("input_dims must contain at least one modality")

        self.hidden_dim = hidden_dim
        self.projections = nn.ModuleDict(
            {name: nn.Linear(dim, hidden_dim, bias=bias) for name, dim in input_dims.items()}
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = activation

    def forward(self, inputs: MutableMapping[str, Tensor]) -> Dict[str, Tensor]:
        """Project each modality tensor into the shared space.

        Args:
            inputs: Mapping of modality name to tensor of shape ``(..., dim)`` where
                ``dim`` matches the declaration in ``input_dims``.

        Returns:
            Mapping of modality name to projected tensor with trailing dimension
            ``hidden_dim``.
        """

        outputs: Dict[str, Tensor] = {}
        for name, tensor in inputs.items():
            if name not in self.projections:
                raise KeyError(f"Unknown modality '{name}' passed to SharedProjection")
            projected = self.projections[name](tensor)
            if self.activation is not None:
                projected = self.activation(projected)
            outputs[name] = self.norm(projected)
        return outputs


class CrossModalAttention(nn.Module):
    """Multi-head attention module across modality representations."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "concat"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'concat'")
        self.reduction = reduction
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=bias, batch_first=True)

    def forward(
        self,
        query: Tensor,
        context: MutableMapping[str, Tensor],
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Attend from ``query`` to each modality in ``context``.

        Args:
            query: Tensor of shape ``(batch, q_len, embed_dim)``.
            context: Mapping of modality name to tensor of shape
                ``(batch, k_len, embed_dim)``.
            attn_mask: Optional attention mask broadcastable to the attention
                logits.
            key_padding_mask: Optional mask identifying padding positions in the
                key/value sequences.

        Returns:
            A tuple ``(fused, weights)`` where ``fused`` is the aggregated
            representation computed according to ``self.reduction`` and
            ``weights`` is a mapping of modality name to the attention weights of
            shape ``(batch, q_len, k_len)``.
        """

        outputs: List[Tensor] = []
        weights: Dict[str, Tensor] = {}
        for name, tensor in context.items():
            out, attn_weights = self.attn(query, tensor, tensor, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            outputs.append(out)
            weights[name] = attn_weights

        if not outputs:
            raise ValueError("context must contain at least one modality to attend to")

        if self.reduction == "mean":
            fused = torch.stack(outputs, dim=0).mean(dim=0)
        elif self.reduction == "sum":
            fused = torch.stack(outputs, dim=0).sum(dim=0)
        else:  # concat
            fused = torch.cat(outputs, dim=-1)
        return fused, weights


class ModalityDropout(nn.Module):
    """Drop entire modality streams during training for robustness."""

    def __init__(self, drop_probability: float = 0.2, min_modalities: int = 1) -> None:
        super().__init__()
        if not 0.0 <= drop_probability < 1.0:
            raise ValueError("drop_probability must be in the range [0, 1)")
        if min_modalities <= 0:
            raise ValueError("min_modalities must be positive")
        self.drop_probability = drop_probability
        self.min_modalities = min_modalities

    def forward(self, inputs: MutableMapping[str, Tensor]) -> Dict[str, Tensor]:
        if not self.training or self.drop_probability == 0.0:
            return dict(inputs)

        modalities = list(inputs.keys())
        if len(modalities) <= self.min_modalities:
            return dict(inputs)

        keep_mask = torch.rand(len(modalities)) > self.drop_probability
        keep_indices = [i for i, keep in enumerate(keep_mask.tolist()) if keep]

        if len(keep_indices) < self.min_modalities:
            # randomly choose modalities to reinstate to satisfy the minimum.
            missing = self.min_modalities - len(keep_indices)
            dropped = [i for i in range(len(modalities)) if i not in keep_indices]
            if missing > len(dropped):
                keep_indices = list(range(len(modalities)))
            else:
                restore = torch.randperm(len(dropped))[:missing].tolist()
                keep_indices.extend(dropped[i] for i in restore)

        kept_modalities = {modalities[i]: inputs[modalities[i]] for i in sorted(set(keep_indices))}
        if not kept_modalities:
            # Fallback to at least one modality.
            kept_modalities[modalities[0]] = inputs[modalities[0]]
        return kept_modalities
