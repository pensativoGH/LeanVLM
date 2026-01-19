"""STEM Language Model.

Full language model with STEM-modified FFN layers. Based on SmolLM2-360M architecture
with independent STEM embedding tables per layer.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stem_decoder import STEMDecoderLayer, RMSNorm

logger = logging.getLogger(__name__)


@dataclass
class STEMConfig:
    """Configuration for STEM Language Model.

    Default values match SmolLM2-360M-Instruct architecture.
    """
    vocab_size: int = 49152
    hidden_size: int = 960
    intermediate_size: int = 2560
    num_hidden_layers: int = 32
    num_attention_heads: int = 15
    num_key_value_heads: int = 5
    max_position_embeddings: int = 8192
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    tie_word_embeddings: bool = True
    # STEM-specific
    stem_init_std: float = 0.02
    num_image_tokens: int = 144


@dataclass
class CausalLMOutput:
    """Output from causal language model forward pass."""
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None
    hidden_states: Optional[torch.Tensor] = None


class STEMLanguageModel(nn.Module):
    """Language model with STEM-modified FFN layers.

    Architecture matches SmolLM2-360M:
    - 32 layers
    - hidden_size=960
    - intermediate_size=2560
    - vocab_size=49152
    - GQA with 15 heads, 5 KV heads

    STEM modification:
    - Each layer has independent stem_embeddings [vocab_size, intermediate_size]
    - Text tokens use embedding lookup instead of up_proj
    - Image tokens use dense up_proj

    Args:
        config: Model configuration
    """

    def __init__(self, config: STEMConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Decoder layers with STEM MLP
        self.layers = nn.ModuleList([
            STEMDecoderLayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                vocab_size=config.vocab_size,
                max_position_embeddings=config.max_position_embeddings,
                rope_theta=config.rope_theta,
                rms_norm_eps=config.rms_norm_eps,
                attention_dropout=config.attention_dropout,
                hidden_act=config.hidden_act,
                stem_init_std=config.stem_init_std,
            )
            for _ in range(config.num_hidden_layers)
        ])

        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Output head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights if specified
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights following standard transformer practice.

        Note: STEM embeddings are initialized separately in STEMMLP with
        config.stem_init_std, so we skip them here to preserve that initialization.
        """
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Skip STEM embeddings - they're initialized in STEMMLP with stem_init_std
            # Only initialize embed_tokens here
            if module is self.embed_tokens:
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        token_ids: Optional[torch.LongTensor] = None,
        num_image_tokens: int = 144,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[torch.LongTensor] = None,
        return_hidden_states: bool = False,
    ) -> CausalLMOutput:
        """Forward pass with STEM token ID propagation.

        Args:
            input_ids: Input token IDs [batch_size, seq_len] (for text-only)
            inputs_embeds: Pre-computed embeddings [batch_size, seq_len, hidden_size]
                          (for multimodal with concatenated image+text embeddings)
            token_ids: Original text token IDs for STEM lookup [batch_size, text_seq_len]
                      Required for STEM when using inputs_embeds
            num_image_tokens: Number of image tokens at sequence start
            attention_mask: Attention mask [batch_size, seq_len]
            position_ids: Position IDs for RoPE
            past_key_values: Cached KV states from previous forward
            use_cache: Whether to return KV cache
            labels: Labels for language modeling loss [batch_size, seq_len]
            return_hidden_states: Whether to return final hidden states

        Returns:
            CausalLMOutput with loss, logits, and optional cache/hidden states
        """
        # Get input embeddings
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided")
            inputs_embeds = self.embed_tokens(input_ids)
            # For text-only, token_ids = input_ids
            if token_ids is None:
                token_ids = input_ids
                num_image_tokens = 0
        else:
            # inputs_embeds provided - validate token_ids for STEM
            seq_len_check = inputs_embeds.shape[1]
            if token_ids is None and seq_len_check > num_image_tokens:
                logger.warning(
                    f"inputs_embeds provided with {seq_len_check - num_image_tokens} text tokens, "
                    "but token_ids is None. STEM embedding lookup will be disabled for text tokens, "
                    "falling back to dense computation. Pass token_ids to enable STEM."
                )

        batch_size, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device

        # Create position IDs
        if position_ids is None:
            past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            position_ids = torch.arange(
                past_len, past_len + seq_len, device=device
            ).unsqueeze(0).expand(batch_size, -1)

        # Create causal attention mask
        if attention_mask is not None:
            # Convert from [batch_size, seq_len] to 4D causal mask
            attention_mask = self._prepare_attention_mask(
                attention_mask, inputs_embeds, past_key_values
            )

        # Forward through decoder layers
        hidden_states = inputs_embeds
        new_past_key_values = () if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None

            hidden_states, new_past_kv = layer(
                hidden_states=hidden_states,
                token_ids=token_ids,
                num_image_tokens=num_image_tokens,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=use_cache,
            )

            if use_cache:
                new_past_key_values = new_past_key_values + (new_past_kv,)

        # Final norm
        hidden_states = self.norm(hidden_states)

        # Compute logits
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=new_past_key_values,
            hidden_states=hidden_states if return_hidden_states else None,
        )

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    ) -> torch.Tensor:
        """Prepare 4D causal attention mask.

        Args:
            attention_mask: [batch_size, seq_len] padding mask
            inputs_embeds: Input embeddings for shape info
            past_key_values: Cached KV for length computation

        Returns:
            4D attention mask [batch_size, 1, seq_len, total_len]
        """
        batch_size, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        # Compute total sequence length including cache
        past_len = 0
        if past_key_values is not None and past_key_values[0] is not None:
            past_len = past_key_values[0][0].shape[2]
        total_len = past_len + seq_len

        # Create causal mask
        causal_mask = torch.triu(
            torch.full((seq_len, total_len), float("-inf"), device=device, dtype=dtype),
            diagonal=past_len + 1,
        )

        # Expand to batch dimension
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, total_len]
        causal_mask = causal_mask.expand(batch_size, 1, seq_len, total_len)

        # Apply padding mask if provided
        if attention_mask is not None:
            # Extend attention mask for past positions
            if past_len > 0:
                attention_mask = torch.cat([
                    torch.ones(batch_size, past_len, device=device, dtype=attention_mask.dtype),
                    attention_mask,
                ], dim=1)

            # Convert to additive mask
            # Use torch.where to avoid 0 * -inf = NaN
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, total_len]
            padding_mask = torch.where(
                padding_mask == 0,
                torch.tensor(float("-inf"), device=device, dtype=dtype),
                torch.tensor(0.0, device=device, dtype=dtype),
            )
            causal_mask = causal_mask + padding_mask

        return causal_mask

    def _get_last_valid_logits(
        self,
        logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Get logits from the last valid (non-padding) position for each sample.

        This is critical for batched generation where samples have different lengths
        and are right-padded. Without this, [:, -1, :] would return logits from
        padding positions for shorter samples.

        Args:
            logits: [batch_size, seq_len, vocab_size]
            attention_mask: [batch_size, seq_len] where 1=valid, 0=padding

        Returns:
            [batch_size, vocab_size] logits from last valid position per sample
        """
        if attention_mask is None:
            # No padding, just use last position
            return logits[:, -1, :]

        batch_size = logits.shape[0]
        device = logits.device

        # Find last valid position: sum of 1s gives count, subtract 1 for 0-indexing
        # attention_mask[i].sum() = number of valid tokens for sample i
        seq_lengths = attention_mask.sum(dim=1).long()  # [batch_size]
        last_valid_idx = (seq_lengths - 1).clamp(min=0)  # [batch_size]

        # Gather logits from last valid positions using advanced indexing
        batch_indices = torch.arange(batch_size, device=device)
        last_logits = logits[batch_indices, last_valid_idx, :]  # [batch_size, vocab_size]

        return last_logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        token_ids: Optional[torch.LongTensor] = None,
        num_image_tokens: int = 144,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """Generate tokens autoregressively with STEM.

        Args:
            input_ids: Initial token IDs [batch_size, seq_len]
            inputs_embeds: Pre-computed embeddings (for multimodal)
            token_ids: Text token IDs for STEM lookup
            num_image_tokens: Number of image tokens
            attention_mask: Input attention mask (combined mask for VLM: image + text)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            do_sample: Whether to sample or use greedy decoding
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID

        Returns:
            Generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        # Handle inputs
        if inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]
            device = inputs_embeds.device
            # First forward with embeddings
            outputs = self.forward(
                inputs_embeds=inputs_embeds,
                token_ids=token_ids,
                num_image_tokens=num_image_tokens,
                attention_mask=attention_mask,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            # Use attention mask to get logits from last valid position (handles padding)
            next_token_logits = self._get_last_valid_logits(outputs.logits, attention_mask)
            # Track generated tokens
            generated_ids = token_ids.clone() if token_ids is not None else torch.empty(
                batch_size, 0, dtype=torch.long, device=device
            )
        else:
            batch_size = input_ids.shape[0]
            device = input_ids.device
            generated_ids = input_ids.clone()
            # First forward
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            # Use attention mask to get logits from last valid position (handles padding)
            next_token_logits = self._get_last_valid_logits(outputs.logits, attention_mask)
            num_image_tokens = 0  # No image tokens for text-only

        # Generate loop
        for _ in range(max_new_tokens):
            # Sample or greedy decode
            if do_sample and temperature > 0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                # Top-p sampling
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_mask = cumsum_probs - sorted_probs > top_p
                    sorted_probs[sorted_mask] = 0.0
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    next_token = torch.gather(
                        sorted_indices, -1,
                        torch.multinomial(sorted_probs, 1)
                    )
                else:
                    next_token = torch.multinomial(probs, 1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            # Append to generated
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # Forward with new token
            # For STEM: next_token is the token_id for lookup
            outputs = self.forward(
                input_ids=next_token,
                token_ids=next_token,  # Single token for STEM lookup
                num_image_tokens=0,  # No image tokens in continuation
                attention_mask=None,  # Not needed with KV cache
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

        return generated_ids

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        config: Optional[STEMConfig] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "STEMLanguageModel":
        """Load from pretrained SmolLM2 checkpoint with STEM modifications.

        This loads the base model weights and initializes STEM embeddings.
        The up_proj weights from the original model are used to initialize
        the image_up_proj in STEM layers.

        Args:
            model_name_or_path: HuggingFace model name or local path
            config: Override config (uses pretrained config if None)
            device: Device to load model on
            dtype: Data type for model weights

        Returns:
            STEMLanguageModel with pretrained weights
        """
        from transformers import AutoModelForCausalLM, AutoConfig

        # Load pretrained config
        hf_config = AutoConfig.from_pretrained(model_name_or_path)

        # Create STEM config from HF config
        if config is None:
            config = STEMConfig(
                vocab_size=hf_config.vocab_size,
                hidden_size=hf_config.hidden_size,
                intermediate_size=hf_config.intermediate_size,
                num_hidden_layers=hf_config.num_hidden_layers,
                num_attention_heads=hf_config.num_attention_heads,
                num_key_value_heads=hf_config.num_key_value_heads,
                max_position_embeddings=hf_config.max_position_embeddings,
                rope_theta=getattr(hf_config, "rope_theta", 10000.0),
                rms_norm_eps=hf_config.rms_norm_eps,
                hidden_act=hf_config.hidden_act,
                tie_word_embeddings=hf_config.tie_word_embeddings,
            )

        # Create STEM model
        model = cls(config)

        # Load pretrained weights
        pretrained = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map="cpu",  # Load to CPU first
        )

        # Copy weights
        model.embed_tokens.weight.data.copy_(pretrained.model.embed_tokens.weight.data)
        model.norm.weight.data.copy_(pretrained.model.norm.weight.data)

        if not config.tie_word_embeddings:
            model.lm_head.weight.data.copy_(pretrained.lm_head.weight.data)

        # Copy layer weights
        for i, (stem_layer, pt_layer) in enumerate(zip(model.layers, pretrained.model.layers)):
            # Attention weights
            stem_layer.self_attn.q_proj.weight.data.copy_(pt_layer.self_attn.q_proj.weight.data)
            stem_layer.self_attn.k_proj.weight.data.copy_(pt_layer.self_attn.k_proj.weight.data)
            stem_layer.self_attn.v_proj.weight.data.copy_(pt_layer.self_attn.v_proj.weight.data)
            stem_layer.self_attn.o_proj.weight.data.copy_(pt_layer.self_attn.o_proj.weight.data)

            # MLP weights (gate_proj and down_proj)
            stem_layer.mlp.gate_proj.weight.data.copy_(pt_layer.mlp.gate_proj.weight.data)
            stem_layer.mlp.down_proj.weight.data.copy_(pt_layer.mlp.down_proj.weight.data)

            # Copy up_proj to image_up_proj (for image tokens)
            stem_layer.mlp.image_up_proj.weight.data.copy_(pt_layer.mlp.up_proj.weight.data)

            # STEM embeddings are randomly initialized (no pretrained equivalent)

            # Layer norms
            stem_layer.input_layernorm.weight.data.copy_(pt_layer.input_layernorm.weight.data)
            stem_layer.post_attention_layernorm.weight.data.copy_(
                pt_layer.post_attention_layernorm.weight.data
            )

        # Move to device/dtype
        if device is not None:
            model = model.to(device)
        if dtype is not None:
            model = model.to(dtype)

        # Cleanup
        del pretrained

        return model
