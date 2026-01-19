"""STEM MLP Module.

Implements the STEM (Scaling Transformers with Embedding Modules) MLP that
replaces the up-projection with embedding lookup for text tokens while
keeping dense computation for image tokens.

Reference: https://arxiv.org/abs/2601.10639
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

logger = logging.getLogger(__name__)


class STEMMLP(nn.Module):
    """FFN with STEM embedding lookup for text tokens, dense for image tokens.

    Standard FFN: output = down_proj(act(gate_proj(x)) * up_proj(x))

    STEM FFN (hybrid):
    - Image tokens: up_values = up_proj(x)           # Dense matmul
    - Text tokens:  up_values = stem_embeddings[token_ids]  # Lookup

    Then: output = down_proj(act(gate_proj(x)) * up_values)

    Args:
        hidden_size: Input/output dimension (960 for SmolLM2-360M)
        intermediate_size: FFN intermediate dimension (2560 for SmolLM2-360M)
        vocab_size: Vocabulary size for STEM embeddings (49152 for SmolLM2)
        hidden_act: Activation function (default: silu)
        init_std: Standard deviation for STEM embedding initialization
    """

    def __init__(
        self,
        hidden_size: int = 960,
        intermediate_size: int = 2560,
        vocab_size: int = 49152,
        hidden_act: str = "silu",
        init_std: float = 0.02,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size

        # Gate projection (kept dense for all tokens)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

        # Down projection (kept dense for all tokens)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        # STEM embeddings for text tokens (replaces up_proj)
        # Shape: [vocab_size, intermediate_size]
        self.stem_embeddings = nn.Embedding(vocab_size, intermediate_size)

        # Dense up_proj for image tokens only
        self.image_up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

        # Activation function
        if hidden_act == "silu":
            self.act_fn = F.silu
        elif hidden_act == "gelu":
            self.act_fn = F.gelu
        elif hidden_act == "relu":
            self.act_fn = F.relu
        else:
            raise ValueError(f"Unsupported activation: {hidden_act}")

        # Initialize STEM embeddings
        self._init_stem_embeddings(init_std)

        # Flag to enable/disable STEM (for staged training)
        # When False, uses dense computation for all tokens
        self.use_stem = True

    def _init_stem_embeddings(self, std: float):
        """Initialize STEM embeddings with normal distribution."""
        nn.init.normal_(self.stem_embeddings.weight, mean=0.0, std=std)

    def enable_stem(self):
        """Enable STEM embedding lookup for text tokens."""
        self.use_stem = True

    def disable_stem(self):
        """Disable STEM, use dense computation for all tokens."""
        self.use_stem = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        num_image_tokens: int = 144,
    ) -> torch.Tensor:
        """Forward pass with hybrid STEM/dense processing.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            token_ids: Original token IDs for STEM lookup [batch_size, text_seq_len]
                      Only needed for text tokens (positions num_image_tokens onwards)
            num_image_tokens: Number of image tokens at the start of sequence

        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute gate values for all tokens (same for both paths)
        gate_values = self.act_fn(self.gate_proj(hidden_states))

        # Compute up-projection values
        if not self.use_stem:
            # STEM disabled: use dense computation for all tokens
            up_values = self.image_up_proj(hidden_states)
        elif num_image_tokens > 0 and seq_len > num_image_tokens:
            # Hybrid case: both image and text tokens present
            up_values = self._hybrid_up_proj(
                hidden_states, token_ids, num_image_tokens
            )
        elif num_image_tokens > 0 and seq_len <= num_image_tokens:
            # Only image tokens (edge case during some forward passes)
            up_values = self.image_up_proj(hidden_states)
        elif token_ids is not None:
            # Only text tokens (e.g., text-only training or generation)
            up_values = self.stem_embeddings(token_ids)
        else:
            # Fallback to dense for backward compatibility
            up_values = self.image_up_proj(hidden_states)

        # Gated output
        output = gate_values * up_values

        # Down projection
        output = self.down_proj(output)

        return output

    def _hybrid_up_proj(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor],
        num_image_tokens: int,
    ) -> torch.Tensor:
        """Compute up-projection with hybrid image/text processing.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            token_ids: [batch_size, text_seq_len] - IDs for text tokens only
            num_image_tokens: Number of image tokens at sequence start

        Returns:
            up_values: [batch_size, seq_len, intermediate_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Pre-allocate output tensor
        up_values = torch.empty(
            batch_size, seq_len, self.intermediate_size,
            device=device, dtype=dtype
        )

        # Image tokens: positions [0, num_image_tokens)
        image_hidden = hidden_states[:, :num_image_tokens, :]
        up_values[:, :num_image_tokens, :] = self.image_up_proj(image_hidden)

        # Text tokens: positions [num_image_tokens, seq_len)
        if token_ids is not None and seq_len > num_image_tokens:
            # STEM embedding lookup for text tokens
            # token_ids should be [batch_size, text_seq_len] where text_seq_len = seq_len - num_image_tokens
            text_up_values = self.stem_embeddings(token_ids)
            up_values[:, num_image_tokens:, :] = text_up_values.to(dtype)
        elif seq_len > num_image_tokens:
            # Fallback: use dense projection if no token_ids provided
            # WARNING: This disables STEM for text tokens!
            logger.warning(
                "STEM fallback: token_ids not provided for text tokens. "
                "Using dense up_proj instead of STEM embeddings. "
                "Pass token_ids to enable STEM for text tokens."
            )
            text_hidden = hidden_states[:, num_image_tokens:, :]
            up_values[:, num_image_tokens:, :] = self.image_up_proj(text_hidden)

        return up_values


class StandardMLP(nn.Module):
    """Standard MLP for comparison/baseline.

    Standard LLaMA-style FFN: output = down_proj(act(gate_proj(x)) * up_proj(x))
    """

    def __init__(
        self,
        hidden_size: int = 960,
        intermediate_size: int = 2560,
        hidden_act: str = "silu",
    ):
        super().__init__()

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        if hidden_act == "silu":
            self.act_fn = F.silu
        elif hidden_act == "gelu":
            self.act_fn = F.gelu
        else:
            self.act_fn = F.silu

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Standard FFN forward pass."""
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )
