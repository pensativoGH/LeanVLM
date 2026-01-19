"""MLP Projector to map vision embeddings to language model space."""

import torch
import torch.nn as nn


class Projector(nn.Module):
    """
    Two-layer MLP projector that maps vision encoder outputs to LM embedding space.

    With pixel shuffle, input is 3072-dim (768 * 4 from 2x2 spatial merge).
    Output dimension matches the language model's hidden size (576 for 135M, 960 for 360M).
    Architecture: Linear(vision_dim -> lm_dim) -> GELU -> Linear(lm_dim -> lm_dim)
    """

    def __init__(
        self,
        vision_hidden_size: int = 3072,  # 768 * 4 after pixel shuffle
        lm_hidden_size: int = 576,
    ):
        super().__init__()

        self.vision_hidden_size = vision_hidden_size
        self.lm_hidden_size = lm_hidden_size

        # Two-layer MLP with GELU activation
        self.mlp = nn.Sequential(
            nn.Linear(vision_hidden_size, lm_hidden_size),
            nn.GELU(),
            nn.Linear(lm_hidden_size, lm_hidden_size),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Project vision features to language model space.

        Args:
            vision_features: (B, num_tokens, 3072) tensor from vision encoder (after pixel shuffle)

        Returns:
            (B, num_tokens, 576) tensor ready for language model
        """
        return self.mlp(vision_features)
