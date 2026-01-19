"""Multimodal Projector Module.

Projects vision encoder outputs to language model hidden dimension.
Supports multiple projector types: linear, mlp, and multi-layer mlp.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalProjector(nn.Module):
    """Projects vision features to language model dimension.

    Supports different projector architectures:
    - linear: Single linear layer
    - mlp: Two-layer MLP with GELU activation
    - multi_layer_mlp: Multi-layer MLP with configurable depth

    Args:
        vision_hidden_size: Input dimension from vision encoder (4608 for SigLIP + pixel shuffle)
        lm_hidden_size: Output dimension for language model (960 for SmolLM2-360M)
        projector_type: Type of projector ('linear', 'mlp', 'multi_layer_mlp')
        num_layers: Number of layers for multi_layer_mlp (default: 2)
        dropout: Dropout probability (default: 0.0)
    """

    def __init__(
        self,
        vision_hidden_size: int = 4608,
        lm_hidden_size: int = 960,
        projector_type: str = "mlp",
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.lm_hidden_size = lm_hidden_size
        self.projector_type = projector_type

        if projector_type == "linear":
            self.projector = nn.Linear(vision_hidden_size, lm_hidden_size)

        elif projector_type == "mlp":
            # Standard 2-layer MLP with GELU
            self.projector = nn.Sequential(
                nn.Linear(vision_hidden_size, lm_hidden_size),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(lm_hidden_size, lm_hidden_size),
            )

        elif projector_type == "multi_layer_mlp":
            # Multi-layer MLP with configurable depth
            layers = []
            in_dim = vision_hidden_size

            for i in range(num_layers - 1):
                layers.extend([
                    nn.Linear(in_dim, lm_hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                ])
                in_dim = lm_hidden_size

            layers.append(nn.Linear(lm_hidden_size, lm_hidden_size))
            self.projector = nn.Sequential(*layers)

        else:
            raise ValueError(f"Unknown projector type: {projector_type}")

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize projector weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """Project vision features to language model dimension.

        Args:
            vision_features: [batch_size, num_patches, vision_hidden_size]

        Returns:
            Projected features [batch_size, num_patches, lm_hidden_size]
        """
        return self.projector(vision_features)


class PerceiverProjector(nn.Module):
    """Perceiver-style projector with cross-attention.

    Uses learned queries to compress visual information through
    cross-attention, allowing flexible control over output tokens.

    Args:
        vision_hidden_size: Input dimension from vision encoder
        lm_hidden_size: Output dimension for language model
        num_queries: Number of output tokens (default: 144 to match pixel shuffle)
        num_heads: Number of attention heads
        num_layers: Number of perceiver layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        vision_hidden_size: int = 4608,
        lm_hidden_size: int = 960,
        num_queries: int = 144,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Learned queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, lm_hidden_size) * 0.02)

        # Input projection
        self.input_proj = nn.Linear(vision_hidden_size, lm_hidden_size)

        # Cross-attention layers
        self.layers = nn.ModuleList([
            PerceiverLayer(lm_hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(lm_hidden_size, lm_hidden_size)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """Project vision features using perceiver architecture.

        Args:
            vision_features: [batch_size, num_patches, vision_hidden_size]

        Returns:
            Projected features [batch_size, num_queries, lm_hidden_size]
        """
        batch_size = vision_features.shape[0]

        # Project input
        kv = self.input_proj(vision_features)

        # Expand queries for batch
        queries = self.queries.expand(batch_size, -1, -1)

        # Apply perceiver layers
        for layer in self.layers:
            queries = layer(queries, kv)

        # Output projection
        return self.output_proj(queries)


class PerceiverLayer(nn.Module):
    """Single perceiver layer with cross-attention and FFN."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm_kv = nn.LayerNorm(hidden_size)

        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(
        self,
        queries: torch.Tensor,
        kv: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with cross-attention.

        Args:
            queries: [batch_size, num_queries, hidden_size]
            kv: [batch_size, num_patches, hidden_size]

        Returns:
            Updated queries [batch_size, num_queries, hidden_size]
        """
        # Cross-attention
        normed_q = self.norm1(queries)
        normed_kv = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(normed_q, normed_kv, normed_kv)
        queries = queries + attn_out

        # FFN
        queries = queries + self.ffn(self.norm2(queries))

        return queries


class ProjectorConfig:
    """Configuration for multimodal projector."""

    def __init__(
        self,
        vision_hidden_size: int = 4608,
        lm_hidden_size: int = 960,
        projector_type: str = "mlp",
        num_layers: int = 2,
        dropout: float = 0.0,
        # Perceiver-specific
        num_queries: int = 144,
        num_heads: int = 8,
    ):
        self.vision_hidden_size = vision_hidden_size
        self.lm_hidden_size = lm_hidden_size
        self.projector_type = projector_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_queries = num_queries
        self.num_heads = num_heads


def build_projector(config: ProjectorConfig) -> nn.Module:
    """Build projector from config.

    Args:
        config: Projector configuration

    Returns:
        Projector module
    """
    if config.projector_type == "perceiver":
        return PerceiverProjector(
            vision_hidden_size=config.vision_hidden_size,
            lm_hidden_size=config.lm_hidden_size,
            num_queries=config.num_queries,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
    else:
        return MultimodalProjector(
            vision_hidden_size=config.vision_hidden_size,
            lm_hidden_size=config.lm_hidden_size,
            projector_type=config.projector_type,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
