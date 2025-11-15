"""
TFAN Multi-Modal Architecture
Vision + Language + Audio fusion with cross-modal attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
import numpy as np

from .attention import RadialSparseAttention
from .topology import DifferentiableTopology


class VisionEncoder(nn.Module):
    """
    Vision encoder with patch embedding and topology.
    Processes images into visual tokens.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        use_topology: bool = True
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Topology (optional)
        self.use_topology = use_topology
        if use_topology:
            self.topology = DifferentiableTopology(
                input_dim=embed_dim,
                max_dim=2,
                num_landmarks=64
            )

        # Transformer layers
        self.layers = nn.ModuleList([
            VisionTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                dropout=0.1
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] images

        Returns:
            Dictionary with:
                - tokens: [B, N+1, D] visual tokens (CLS + patches)
                - cls_token: [B, D] classification token
                - topology: Topological features (optional)
        """
        B = x.size(0)

        # Patch embedding: [B, C, H, W] -> [B, D, H/P, W/P] -> [B, D, N] -> [B, N, D]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # Extract features
        cls_token = x[:, 0]
        patch_tokens = x[:, 1:]

        result = {
            'tokens': x,
            'cls_token': cls_token,
            'patch_tokens': patch_tokens
        }

        # Compute topology
        if self.use_topology:
            topo_features = self.topology(patch_tokens)
            result['topology'] = topo_features

        return result


class VisionTransformerBlock(nn.Module):
    """Vision Transformer block with attention and MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


class LanguageEncoder(nn.Module):
    """
    Language encoder with token embedding and sparse attention.
    Processes text into linguistic tokens.
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        embed_dim: int = 768,
        max_seq_len: int = 512,
        num_layers: int = 6,
        num_heads: int = 12,
        use_sparse_attention: bool = True
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_seq_len, embed_dim) * 0.02
        )

        # Transformer layers
        self.use_sparse_attention = use_sparse_attention
        self.layers = nn.ModuleList([
            LanguageTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                use_sparse_attention=use_sparse_attention,
                dropout=0.1
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [B, L] token IDs
            attention_mask: [B, L] attention mask (optional)

        Returns:
            Dictionary with:
                - tokens: [B, L, D] linguistic tokens
                - pooled: [B, D] pooled representation
        """
        B, L = input_ids.size()

        # Token embedding
        x = self.token_embed(input_ids)

        # Add positional embedding
        x = x + self.pos_embed[:, :L, :]

        # Transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.norm(x)

        # Pooling (mean over sequence)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(x)
            sum_embeddings = (x * mask_expanded).sum(1)
            sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = x.mean(dim=1)

        return {
            'tokens': x,
            'pooled': pooled
        }


class LanguageTransformerBlock(nn.Module):
    """Language Transformer block with optional sparse attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        use_sparse_attention: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.use_sparse_attention = use_sparse_attention
        if use_sparse_attention:
            self.attn = RadialSparseAttention(
                embed_dim=dim,
                num_heads=num_heads,
                num_landmarks=64,
                keep_ratio=0.3
            )
        else:
            self.attn = nn.MultiheadAttention(
                dim,
                num_heads,
                dropout=dropout,
                batch_first=True
            )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        if self.use_sparse_attention:
            x = x + self.attn(self.norm1(x))
        else:
            x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


class AudioEncoder(nn.Module):
    """
    Audio encoder with spectrogram processing.
    Processes audio spectrograms into acoustic tokens.
    """

    def __init__(
        self,
        n_mels: int = 80,
        max_frames: int = 3000,
        embed_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 8
    ):
        super().__init__()

        self.n_mels = n_mels
        self.max_frames = max_frames
        self.embed_dim = embed_dim

        # Conv layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(n_mels, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_frames // 8, embed_dim) * 0.02
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            VisionTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                dropout=0.1
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, n_mels, T] mel-spectrogram

        Returns:
            Dictionary with:
                - tokens: [B, T', D] acoustic tokens
                - pooled: [B, D] pooled representation
        """
        # Conv feature extraction
        x = self.conv_layers(x)  # [B, D, T']
        x = x.transpose(1, 2)  # [B, T', D]

        # Add positional embedding
        T = x.size(1)
        x = x + self.pos_embed[:, :T, :]

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # Pooling
        pooled = x.mean(dim=1)

        return {
            'tokens': x,
            'pooled': pooled
        }


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for fusing different modalities.
    Allows each modality to attend to other modalities.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Multi-head cross attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: [B, N_q, D] query tokens (target modality)
            key_value: [B, N_kv, D] key/value tokens (source modality)
            attention_mask: [B, N_q, N_kv] attention mask (optional)

        Returns:
            [B, N_q, D] attended query tokens
        """
        # Cross-attention
        attn_output, _ = self.cross_attn(
            self.norm1(query),
            self.norm1(key_value),
            self.norm1(key_value),
            attn_mask=attention_mask
        )
        query = query + attn_output

        # MLP
        query = query + self.mlp(self.norm2(query))

        return query


class MultiModalFusion(nn.Module):
    """
    Multi-modal fusion module.
    Fuses vision, language, and audio modalities with cross-modal attention.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_fusion_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Cross-modal attention layers
        self.fusion_layers = nn.ModuleList([
            nn.ModuleDict({
                'vision_to_language': CrossModalAttention(embed_dim, num_heads, dropout),
                'language_to_vision': CrossModalAttention(embed_dim, num_heads, dropout),
                'vision_to_audio': CrossModalAttention(embed_dim, num_heads, dropout),
                'audio_to_vision': CrossModalAttention(embed_dim, num_heads, dropout),
                'language_to_audio': CrossModalAttention(embed_dim, num_heads, dropout),
                'audio_to_language': CrossModalAttention(embed_dim, num_heads, dropout),
            })
            for _ in range(num_fusion_layers)
        ])

        # Fusion projection
        self.fusion_proj = nn.Linear(embed_dim * 3, embed_dim)

    def forward(
        self,
        vision_tokens: Optional[torch.Tensor] = None,
        language_tokens: Optional[torch.Tensor] = None,
        audio_tokens: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            vision_tokens: [B, N_v, D] visual tokens (optional)
            language_tokens: [B, N_l, D] linguistic tokens (optional)
            audio_tokens: [B, N_a, D] acoustic tokens (optional)

        Returns:
            Dictionary with:
                - vision_fused: Fused visual features
                - language_fused: Fused linguistic features
                - audio_fused: Fused acoustic features
                - multimodal: Combined multimodal representation
        """
        # Check which modalities are present
        has_vision = vision_tokens is not None
        has_language = language_tokens is not None
        has_audio = audio_tokens is not None

        # Cross-modal fusion
        for layer in self.fusion_layers:
            # Vision ↔ Language
            if has_vision and has_language:
                vision_tokens = layer['vision_to_language'](vision_tokens, language_tokens)
                language_tokens = layer['language_to_vision'](language_tokens, vision_tokens)

            # Vision ↔ Audio
            if has_vision and has_audio:
                vision_tokens = layer['vision_to_audio'](vision_tokens, audio_tokens)
                audio_tokens = layer['audio_to_vision'](audio_tokens, vision_tokens)

            # Language ↔ Audio
            if has_language and has_audio:
                language_tokens = layer['language_to_audio'](language_tokens, audio_tokens)
                audio_tokens = layer['audio_to_language'](audio_tokens, language_tokens)

        # Pooling and fusion
        pooled = []
        result = {}

        if has_vision:
            vision_pooled = vision_tokens.mean(dim=1)
            pooled.append(vision_pooled)
            result['vision_fused'] = vision_tokens
        else:
            pooled.append(torch.zeros(1, self.embed_dim, device=vision_tokens.device if has_vision else language_tokens.device))

        if has_language:
            language_pooled = language_tokens.mean(dim=1)
            pooled.append(language_pooled)
            result['language_fused'] = language_tokens
        else:
            pooled.append(torch.zeros(1, self.embed_dim, device=language_tokens.device if has_language else vision_tokens.device))

        if has_audio:
            audio_pooled = audio_tokens.mean(dim=1)
            pooled.append(audio_pooled)
            result['audio_fused'] = audio_tokens
        else:
            pooled.append(torch.zeros(1, self.embed_dim, device=audio_tokens.device if has_audio else vision_tokens.device))

        # Combine modalities
        multimodal = self.fusion_proj(torch.cat(pooled, dim=-1))
        result['multimodal'] = multimodal

        return result


class TFANMultiModal(nn.Module):
    """
    TFAN Multi-Modal Model
    Complete architecture for vision + language + audio processing.
    """

    def __init__(
        self,
        # Vision
        img_size: int = 224,
        patch_size: int = 16,
        # Language
        vocab_size: int = 50000,
        max_seq_len: int = 512,
        # Audio
        n_mels: int = 80,
        max_audio_frames: int = 3000,
        # Shared
        embed_dim: int = 768,
        num_heads: int = 12,
        # Architecture
        vision_layers: int = 6,
        language_layers: int = 6,
        audio_layers: int = 4,
        fusion_layers: int = 3,
        # Task
        num_classes: int = 1000,
        task: str = "classification"  # classification, retrieval, generation
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.task = task

        # Modality encoders
        self.vision_encoder = VisionEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_layers=vision_layers,
            num_heads=num_heads
        )

        self.language_encoder = LanguageEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
            num_layers=language_layers,
            num_heads=num_heads
        )

        self.audio_encoder = AudioEncoder(
            n_mels=n_mels,
            max_frames=max_audio_frames,
            embed_dim=embed_dim,
            num_layers=audio_layers,
            num_heads=num_heads
        )

        # Multi-modal fusion
        self.fusion = MultiModalFusion(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_fusion_layers=fusion_layers
        )

        # Task-specific heads
        if task == "classification":
            self.classifier = nn.Linear(embed_dim, num_classes)
        elif task == "retrieval":
            self.projection = nn.Linear(embed_dim, embed_dim)
        elif task == "generation":
            self.decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 4,
                    batch_first=True
                ),
                num_layers=6
            )
            self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: [B, C, H, W] images (optional)
            input_ids: [B, L] token IDs (optional)
            audio: [B, n_mels, T] mel-spectrograms (optional)
            attention_mask: [B, L] attention mask (optional)

        Returns:
            Dictionary with task-specific outputs
        """
        # Encode modalities
        vision_output = None
        if images is not None:
            vision_output = self.vision_encoder(images)
            vision_tokens = vision_output['tokens']
        else:
            vision_tokens = None

        language_output = None
        if input_ids is not None:
            language_output = self.language_encoder(input_ids, attention_mask)
            language_tokens = language_output['tokens']
        else:
            language_tokens = None

        audio_output = None
        if audio is not None:
            audio_output = self.audio_encoder(audio)
            audio_tokens = audio_output['tokens']
        else:
            audio_tokens = None

        # Multi-modal fusion
        fusion_output = self.fusion(
            vision_tokens=vision_tokens,
            language_tokens=language_tokens,
            audio_tokens=audio_tokens
        )

        multimodal_repr = fusion_output['multimodal']

        # Task-specific processing
        result = {
            'vision_output': vision_output,
            'language_output': language_output,
            'audio_output': audio_output,
            'fusion_output': fusion_output,
            'multimodal_repr': multimodal_repr
        }

        if self.task == "classification":
            logits = self.classifier(multimodal_repr)
            result['logits'] = logits

        elif self.task == "retrieval":
            embeddings = F.normalize(self.projection(multimodal_repr), dim=-1)
            result['embeddings'] = embeddings

        return result
