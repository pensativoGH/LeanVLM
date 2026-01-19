"""Combined Vision-Language Model."""

import torch
import torch.nn as nn
from typing import Optional

from .vision_encoder import VisionEncoder
from .projector import Projector
from .language_model import LanguageModel


class TinyVLM(nn.Module):
    """
    TinyVLM: A small Vision-Language Model.

    Architecture:
        Image (384x384) -> VisionEncoder (SigLIP) -> 576 patches (24x24)
                       -> PixelShuffle (2x2) -> 144 tokens (3072-dim)
                       -> Projector -> 144 tokens (576-dim)
                       -> Concat with text embeddings
                       -> LanguageModel -> Output
    """

    def __init__(
        self,
        vision_encoder_name: str = "google/siglip-base-patch16-384",
        language_model_name: str = "HuggingFaceTB/SmolLM2-135M",
        image_size: int = 384,  # SigLIP native size
        num_image_tokens: int = 144,  # 12x12 after pixel shuffle
        freeze_vision: bool = True,
        freeze_lm: bool = False,
    ):
        super().__init__()

        self.image_size = image_size
        self.num_image_tokens = num_image_tokens

        # Initialize components
        self.vision_encoder = VisionEncoder(
            model_name=vision_encoder_name,
            image_size=image_size,
            num_output_tokens=num_image_tokens,
            freeze=freeze_vision,
        )

        self.language_model = LanguageModel(
            model_name=language_model_name,
            freeze=freeze_lm,
        )

        self.projector = Projector(
            vision_hidden_size=self.vision_encoder.hidden_size,
            lm_hidden_size=self.language_model.hidden_size,
        )

        # Store dimensions
        self.vision_hidden_size = self.vision_encoder.hidden_size
        self.lm_hidden_size = self.language_model.hidden_size

    def freeze_vision_encoder(self):
        """Freeze the vision encoder."""
        self.vision_encoder.freeze()

    def unfreeze_vision_encoder(self):
        """Unfreeze the vision encoder for fine-tuning."""
        self.vision_encoder.unfreeze()

    def freeze_language_model(self):
        """Freeze the language model."""
        self.language_model.freeze()

    def unfreeze_language_model(self):
        """Unfreeze the language model."""
        self.language_model.unfreeze()

    def freeze_language_model_partial(self, unfreeze_layers: list = None):
        """
        Partially freeze the language model.

        Useful for Stage 1 training where we want to train embeddings + first block
        while keeping the rest of the LM frozen.

        Args:
            unfreeze_layers: List of layers to keep trainable.
                            - "embed": embedding layer
                            - 0, 1, 2, ...: transformer layer indices
                            Default: ["embed", 0] (embeddings + first block)
        """
        print("Partial LM freeze:")
        self.language_model.freeze_partial(unfreeze_layers)

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        """
        Resize token embeddings to accommodate new tokens.

        This is used when adding special tokens (e.g., <think>, </think>)
        for post-training.

        Args:
            new_num_tokens: New vocabulary size
        """
        print(f"Resizing model embeddings: {self.language_model.vocab_size} -> {new_num_tokens}")
        self.language_model.resize_token_embeddings(new_num_tokens)

    def get_trainable_parameters(self):
        """Get all trainable parameters."""
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                params.append(param)
        return params

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass through the VLM.

        Args:
            pixel_values: (B, 3, 384, 384) normalized images
            input_ids: (B, text_seq_len) text token IDs
            attention_mask: (B, total_seq_len) attention mask for full sequence
            labels: (B, total_seq_len) labels for loss (-100 for ignored positions)

        Returns:
            CausalLMOutput with loss and logits
        """
        batch_size = pixel_values.shape[0]

        # Encode images: (B, 144, 3072) after pixel shuffle
        vision_features = self.vision_encoder(pixel_values)

        # Project to LM space: (B, 144, 576)
        image_embeds = self.projector(vision_features)

        # Get text embeddings: (B, text_seq_len, 576)
        text_embeds = self.language_model.embed_tokens(input_ids)

        # Concatenate: [image_tokens, text_tokens]
        # (B, 144 + text_seq_len, 576)
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        # Forward through language model
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: int = None,
        eos_token_id: int = None,
    ):
        """
        Generate text given an image and prompt.

        Args:
            pixel_values: (B, 3, 384, 384) normalized images
            input_ids: (B, text_seq_len) text token IDs (prompt)
            attention_mask: (B, 144 + text_seq_len) attention mask
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            do_sample: Whether to sample or use greedy decoding
            pad_token_id: Padding token ID
            eos_token_id: End of sequence token ID

        Returns:
            Generated token IDs (excluding image tokens and prompt)
        """
        # Get target dtype from language model (handles mixed precision)
        target_dtype = next(self.language_model.parameters()).dtype

        # Convert pixel_values to target dtype for consistent processing
        pixel_values = pixel_values.to(target_dtype)

        # Encode images: (B, 144, 3072) after pixel shuffle
        vision_features = self.vision_encoder(pixel_values)

        # Project to LM space: (B, 144, 576) - ensure correct dtype
        image_embeds = self.projector(vision_features).to(target_dtype)

        # Get text embeddings: (B, text_seq_len, 576)
        text_embeds = self.language_model.embed_tokens(input_ids)

        # Concatenate: [image_tokens, text_tokens]
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        # Generate
        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

        return outputs

    def save_pretrained(self, save_path: str):
        """Save the model checkpoint."""
        torch.save({
            'vision_encoder': self.vision_encoder.state_dict(),
            'projector': self.projector.state_dict(),
            'language_model': self.language_model.model.state_dict(),
            'config': {
                'image_size': self.image_size,
                'num_image_tokens': self.num_image_tokens,
                'vision_hidden_size': self.vision_hidden_size,
                'lm_hidden_size': self.lm_hidden_size,
            }
        }, save_path)

    def load_pretrained(self, load_path: str, strict: bool = True):
        """Load a model checkpoint."""
        checkpoint = torch.load(load_path, map_location='cpu')
        self.vision_encoder.load_state_dict(checkpoint['vision_encoder'], strict=strict)
        self.projector.load_state_dict(checkpoint['projector'], strict=strict)
        self.language_model.model.load_state_dict(checkpoint['language_model'], strict=strict)
