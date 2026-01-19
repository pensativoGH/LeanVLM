"""STEM Decoder Layer.

Implements a transformer decoder layer with STEM MLP that propagates
token IDs through the layer for STEM embedding lookup.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stem_mlp import STEMMLP


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Compute inverse frequencies
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cos/sin cache
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Extend cache if needed
        seq_len = position_ids.max().item() + 1
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        cos = self.cos_cached[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = self.sin_cached[position_ids].unsqueeze(1)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class STEMAttention(nn.Module):
    """Multi-head attention with RoPE for STEM decoder.

    Args:
        hidden_size: Hidden dimension
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of KV heads (for GQA)
        max_position_embeddings: Maximum sequence length for RoPE
        rope_theta: Base for RoPE frequency computation
        attention_dropout: Dropout probability for attention weights
    """

    def __init__(
        self,
        hidden_size: int = 960,
        num_attention_heads: int = 15,
        num_key_value_heads: int = 5,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_kv_groups = num_attention_heads // num_key_value_heads
        self.attention_dropout = attention_dropout

        # Projections
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)

        # Rotary embedding
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Expand KV for grouped query attention
        if self.num_kv_groups > 1:
            key_states = key_states.repeat_interleave(self.num_kv_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_kv_groups, dim=1)

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        if self.training and self.attention_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value


class STEMDecoderLayer(nn.Module):
    """Transformer decoder layer with STEM MLP.

    Args:
        hidden_size: Hidden dimension
        intermediate_size: FFN intermediate dimension
        num_attention_heads: Number of attention heads
        num_key_value_heads: Number of KV heads
        vocab_size: Vocabulary size for STEM embeddings
        max_position_embeddings: Max sequence length for RoPE
        rope_theta: Base for RoPE frequencies
        rms_norm_eps: Epsilon for RMSNorm
        attention_dropout: Attention dropout probability
        hidden_act: Activation function
        stem_init_std: STEM embedding initialization std
    """

    def __init__(
        self,
        hidden_size: int = 960,
        intermediate_size: int = 2560,
        num_attention_heads: int = 15,
        num_key_value_heads: int = 5,
        vocab_size: int = 49152,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        hidden_act: str = "silu",
        stem_init_std: float = 0.02,
    ):
        super().__init__()

        # Self attention
        self.self_attn = STEMAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            attention_dropout=attention_dropout,
        )

        # STEM MLP
        self.mlp = STEMMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            hidden_act=hidden_act,
            init_std=stem_init_std,
        )

        # Layer norms
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        num_image_tokens: int = 144,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with STEM MLP.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            token_ids: [batch_size, text_seq_len] - Original token IDs for STEM lookup
            num_image_tokens: Number of image tokens at sequence start
            attention_mask: Attention mask [batch_size, 1, seq_len, seq_len]
            position_ids: Position IDs for RoPE
            past_key_value: Cached KV states
            use_cache: Whether to return KV cache

        Returns:
            hidden_states: Output tensor
            past_key_value: Updated KV cache (if use_cache)
        """
        residual = hidden_states

        # Pre-norm + attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, past_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Pre-norm + STEM MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(
            hidden_states,
            token_ids=token_ids,
            num_image_tokens=num_image_tokens,
        )
        hidden_states = residual + hidden_states

        return hidden_states, past_key_value
