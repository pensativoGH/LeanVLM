"""STEM Vision-Language Model.

Main VLM class that integrates vision encoder, projector, and STEM language model.
Supports multimodal training and generation with STEM embedding lookup for text tokens.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_encoder import VisionEncoder, VisionEncoderConfig
from .projector import MultimodalProjector, ProjectorConfig, build_projector
from .stem_language_model import STEMLanguageModel, STEMConfig, CausalLMOutput


@dataclass
class STEMVLMConfig:
    """Configuration for STEM Vision-Language Model."""

    # Vision encoder config
    vision_model_name: str = "google/siglip-so400m-patch14-384"
    pixel_shuffle_scale: int = 2
    freeze_vision_encoder: bool = True
    image_size: int = 384

    # Projector config
    projector_type: str = "mlp"
    projector_num_layers: int = 2
    projector_dropout: float = 0.0

    # Language model config (SmolLM2-360M defaults)
    lm_model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
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

    # STEM config
    stem_init_std: float = 0.02

    # Computed values (set in __post_init__)
    vision_hidden_size: int = field(default=4608, init=False)
    num_image_tokens: int = field(default=144, init=False)

    def __post_init__(self):
        # Compute vision output dimensions based on vision encoder
        # SigLIP variants have different hidden sizes
        if "so400m" in self.vision_model_name.lower():
            # SigLIP-SO400M: hidden=1152, patch14 → 27x27=729 patches
            base_vision_hidden = 1152
            num_patches = 729
        elif "base" in self.vision_model_name.lower() and "patch16" in self.vision_model_name.lower():
            # SigLIP-base-patch16: hidden=768, patch16 → 24x24=576 patches
            base_vision_hidden = 768
            num_patches = 576
        else:
            # Default to SigLIP-base values
            base_vision_hidden = 768
            num_patches = 576

        self.vision_hidden_size = base_vision_hidden * (self.pixel_shuffle_scale ** 2)
        self.num_image_tokens = num_patches // (self.pixel_shuffle_scale ** 2)


@dataclass
class STEMVLMOutput:
    """Output from STEM VLM forward pass."""
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None
    hidden_states: Optional[torch.Tensor] = None
    image_features: Optional[torch.Tensor] = None


class STEMVLM(nn.Module):
    """Vision-Language Model with STEM language model.

    Architecture:
    - Vision Encoder: SigLIP with pixel shuffle → 144 tokens × 4608-dim
    - Projector: MLP to project to LM hidden size → 144 tokens × 960-dim
    - Language Model: STEM-modified SmolLM2 with embedding lookup for text

    Flow:
    Image → VisionEncoder → Projector → [image_embeds]
    Text → Tokenizer → LM.embed_tokens → [text_embeds]
    Concatenate → [image_embeds | text_embeds] → STEM LM → Output

    STEM modification:
    - Image tokens (positions 0-143): Dense up_proj in FFN
    - Text tokens (positions 144+): STEM embedding lookup in FFN

    Args:
        config: Model configuration
    """

    def __init__(self, config: STEMVLMConfig):
        super().__init__()
        self.config = config

        # Vision encoder
        self.vision_encoder = VisionEncoder(
            model_name=config.vision_model_name,
            pixel_shuffle_scale=config.pixel_shuffle_scale,
            freeze=config.freeze_vision_encoder,
        )

        # Projector
        projector_config = ProjectorConfig(
            vision_hidden_size=config.vision_hidden_size,
            lm_hidden_size=config.hidden_size,
            projector_type=config.projector_type,
            num_layers=config.projector_num_layers,
            dropout=config.projector_dropout,
        )
        self.projector = build_projector(projector_config)

        # STEM Language Model
        lm_config = STEMConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            attention_dropout=config.attention_dropout,
            hidden_act=config.hidden_act,
            tie_word_embeddings=config.tie_word_embeddings,
            stem_init_std=config.stem_init_std,
            num_image_tokens=config.num_image_tokens,
        )
        self.language_model = STEMLanguageModel(lm_config)

        # Store important values
        self.num_image_tokens = config.num_image_tokens

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images to visual features.

        Args:
            pixel_values: Input images [batch_size, 3, 384, 384]

        Returns:
            Projected visual features [batch_size, 144, 960]
        """
        # Vision encoder
        vision_features = self.vision_encoder(pixel_values)

        # Project to LM dimension
        image_embeds = self.projector(vision_features)

        return image_embeds

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        return_image_features: bool = False,
    ) -> STEMVLMOutput:
        """Forward pass for STEM VLM.

        Args:
            pixel_values: Input images [batch_size, 3, 384, 384]
            input_ids: Text token IDs [batch_size, text_seq_len]
            attention_mask: Attention mask for text [batch_size, text_seq_len]
            labels: Labels for LM loss [batch_size, seq_len]
            image_embeds: Pre-computed image embeddings (optional)
            past_key_values: Cached KV states
            use_cache: Whether to return KV cache
            return_image_features: Whether to return image features

        Returns:
            STEMVLMOutput with loss, logits, and optional outputs
        """
        batch_size = input_ids.shape[0] if input_ids is not None else pixel_values.shape[0]
        device = input_ids.device if input_ids is not None else pixel_values.device

        # Encode images if not provided
        if image_embeds is None and pixel_values is not None:
            image_embeds = self.encode_images(pixel_values)
        elif image_embeds is None:
            # Text-only mode
            return self._forward_text_only(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

        # Get text embeddings
        text_embeds = self.language_model.embed_tokens(input_ids)

        # Concatenate: [image_embeds | text_embeds]
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        # Create combined attention mask
        if attention_mask is not None:
            # Image tokens always attend
            image_attention = torch.ones(
                batch_size, self.num_image_tokens,
                device=device, dtype=attention_mask.dtype
            )
            combined_attention_mask = torch.cat([image_attention, attention_mask], dim=1)
        else:
            combined_attention_mask = None

        # Prepare labels: ignore image tokens
        if labels is not None:
            # Prepend -100 for image tokens
            image_labels = torch.full(
                (batch_size, self.num_image_tokens),
                fill_value=-100,
                device=device,
                dtype=labels.dtype,
            )
            combined_labels = torch.cat([image_labels, labels], dim=1)
        else:
            combined_labels = None

        # Forward through STEM language model
        # Pass text token_ids for STEM lookup
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            token_ids=input_ids,  # Original text IDs for STEM lookup
            num_image_tokens=self.num_image_tokens,
            attention_mask=combined_attention_mask,
            labels=combined_labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        return STEMVLMOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            image_features=image_embeds if return_image_features else None,
        )

    def _forward_text_only(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.LongTensor],
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        use_cache: bool,
    ) -> STEMVLMOutput:
        """Forward pass for text-only inputs."""
        outputs = self.language_model(
            input_ids=input_ids,
            token_ids=input_ids,
            num_image_tokens=0,  # No image tokens
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        return STEMVLMOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """Generate text conditioned on image.

        Args:
            pixel_values: Input images [batch_size, 3, 384, 384]
            input_ids: Initial text tokens [batch_size, text_seq_len]
            attention_mask: Text attention mask
            image_embeds: Pre-computed image embeddings
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            do_sample: Whether to sample or greedy decode
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID

        Returns:
            Generated token IDs [batch_size, text_seq_len + max_new_tokens]
        """
        # Encode images if needed
        if image_embeds is None and pixel_values is not None:
            image_embeds = self.encode_images(pixel_values)

        batch_size = input_ids.shape[0] if input_ids is not None else image_embeds.shape[0]
        device = input_ids.device if input_ids is not None else image_embeds.device

        if image_embeds is None:
            # Text-only generation
            return self.language_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )

        # Get text embeddings
        if input_ids is not None:
            text_embeds = self.language_model.embed_tokens(input_ids)
            inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
            token_ids = input_ids
        else:
            inputs_embeds = image_embeds
            token_ids = None

        # Create attention mask
        if attention_mask is not None:
            image_attention = torch.ones(
                batch_size, self.num_image_tokens,
                device=device, dtype=attention_mask.dtype
            )
            combined_mask = torch.cat([image_attention, attention_mask], dim=1)
        else:
            combined_mask = None

        # Generate using language model
        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            token_ids=token_ids,
            num_image_tokens=self.num_image_tokens,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

    def get_input_embeddings(self) -> nn.Embedding:
        """Get input embeddings from language model."""
        return self.language_model.embed_tokens

    @classmethod
    def from_pretrained(
        cls,
        lm_model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
        vision_model_name: str = "google/siglip-so400m-patch14-384",
        config: Optional[STEMVLMConfig] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "STEMVLM":
        """Create STEM VLM from pretrained models.

        Loads SmolLM2 weights into STEM language model and SigLIP for vision.
        STEM embeddings are randomly initialized.

        Args:
            lm_model_name: HuggingFace LM model name
            vision_model_name: HuggingFace vision model name
            config: Optional config override
            device: Target device
            dtype: Target dtype

        Returns:
            STEMVLM with pretrained weights
        """
        # Create config if not provided
        if config is None:
            config = STEMVLMConfig(
                lm_model_name=lm_model_name,
                vision_model_name=vision_model_name,
            )

        # Create model
        model = cls(config)

        # Load pretrained LM weights
        model.language_model = STEMLanguageModel.from_pretrained(
            lm_model_name,
            device=device,
            dtype=dtype,
        )

        # Vision encoder is already loaded from pretrained in __init__

        # Move to device/dtype
        if device is not None:
            model = model.to(device)
        if dtype is not None:
            # Convert non-vision components (vision encoder may want float32)
            model.projector = model.projector.to(dtype)
            model.language_model = model.language_model.to(dtype)

        return model

    def freeze_vision_encoder(self):
        """Freeze vision encoder parameters."""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def unfreeze_vision_encoder(self):
        """Unfreeze vision encoder parameters."""
        for param in self.vision_encoder.parameters():
            param.requires_grad = True

    def freeze_language_model(self):
        """Freeze language model parameters (except STEM embeddings)."""
        for name, param in self.language_model.named_parameters():
            if "stem_embeddings" not in name:
                param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def enable_stem(self):
        """Enable STEM embedding lookup for text tokens in all layers."""
        for layer in self.language_model.layers:
            layer.mlp.enable_stem()

    def disable_stem(self):
        """Disable STEM, use dense computation for all tokens in all layers."""
        for layer in self.language_model.layers:
            layer.mlp.disable_stem()

    def get_num_params(self, trainable_only: bool = False) -> Dict[str, int]:
        """Get number of parameters by component.

        Args:
            trainable_only: Only count trainable parameters

        Returns:
            Dict with parameter counts per component
        """
        def count_params(module: nn.Module, trainable: bool) -> int:
            if trainable:
                return sum(p.numel() for p in module.parameters() if p.requires_grad)
            return sum(p.numel() for p in module.parameters())

        counts = {
            "vision_encoder": count_params(self.vision_encoder, trainable_only),
            "projector": count_params(self.projector, trainable_only),
            "language_model": count_params(self.language_model, trainable_only),
        }

        # Count STEM embeddings separately
        stem_count = 0
        for layer in self.language_model.layers:
            if trainable_only:
                if layer.mlp.stem_embeddings.weight.requires_grad:
                    stem_count += layer.mlp.stem_embeddings.weight.numel()
            else:
                stem_count += layer.mlp.stem_embeddings.weight.numel()
        counts["stem_embeddings"] = stem_count

        counts["total"] = sum(counts.values()) - counts["stem_embeddings"]  # Avoid double count
        counts["total"] = count_params(self, trainable_only)

        return counts
