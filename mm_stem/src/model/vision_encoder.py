"""Vision Encoder Module.

Uses SigLIP for image encoding with pixel shuffle for token reduction.
Produces 144 image tokens from 384x384 input images.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class PixelShuffle(nn.Module):
    """Pixel shuffle for spatial token reduction.

    Reduces spatial dimensions by factor of `scale` while increasing channel dimension.
    Used to reduce the number of image tokens from vision encoder.

    Args:
        scale: Downsampling factor (default: 2 for 4x token reduction)
    """

    def __init__(self, scale: int = 2):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pixel shuffle.

        Args:
            x: Input tensor [batch_size, num_patches, hidden_dim]
               where num_patches = H * W (typically 576 for SigLIP)

        Returns:
            Reduced tensor [batch_size, num_patches // scale^2, hidden_dim * scale^2]
        """
        batch_size, seq_len, hidden_dim = x.shape

        # Compute spatial dimensions (assuming square)
        h = w = int(seq_len ** 0.5)
        assert h * w == seq_len, f"Sequence length {seq_len} is not a perfect square"

        # Reshape to spatial
        x = x.view(batch_size, h, w, hidden_dim)

        # Rearrange: [B, H, W, C] -> [B, H//s, s, W//s, s, C]
        s = self.scale
        x = x.view(batch_size, h // s, s, w // s, s, hidden_dim)

        # Permute and flatten scale dimensions into channel
        # [B, H//s, s, W//s, s, C] -> [B, H//s, W//s, s, s, C] -> [B, H//s, W//s, s*s*C]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(batch_size, h // s, w // s, s * s * hidden_dim)

        # Flatten spatial back to sequence
        x = x.view(batch_size, (h // s) * (w // s), s * s * hidden_dim)

        return x


class VisionEncoder(nn.Module):
    """Vision encoder using SigLIP with pixel shuffle.

    Flow:
    Image (384x384) → SigLIP → [576 patches, 1152-dim]
                    → PixelShuffle (2x) → [144 patches, 4608-dim]

    Args:
        model_name: SigLIP model name from HuggingFace
        pixel_shuffle_scale: Downsampling factor for pixel shuffle
        freeze: Whether to freeze encoder weights
    """

    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        pixel_shuffle_scale: int = 2,
        freeze: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.pixel_shuffle_scale = pixel_shuffle_scale
        self.freeze = freeze

        # Load SigLIP
        self._load_siglip()

        # Pixel shuffle for token reduction
        self.pixel_shuffle = PixelShuffle(scale=pixel_shuffle_scale)

        # Freeze if specified
        if freeze:
            self._freeze_encoder()

    def _load_siglip(self):
        """Load SigLIP vision encoder from HuggingFace."""
        from transformers import SiglipVisionModel

        self.encoder = SiglipVisionModel.from_pretrained(self.model_name)

        # Store dimensions
        self.hidden_size = self.encoder.config.hidden_size  # 1152
        self.image_size = self.encoder.config.image_size  # 384
        self.patch_size = self.encoder.config.patch_size  # 14

        # Compute output dimensions
        patches_per_side = self.image_size // self.patch_size  # 27 -> but SigLIP uses 576 patches = 24x24
        # SigLIP-so400m uses 24x24 = 576 patches
        self.num_patches = 576
        self.num_output_tokens = self.num_patches // (self.pixel_shuffle_scale ** 2)  # 144
        self.output_hidden_size = self.hidden_size * (self.pixel_shuffle_scale ** 2)  # 4608

    def _freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_dict: bool = True,
    ) -> torch.Tensor:
        """Encode images to visual tokens.

        Args:
            pixel_values: Input images [batch_size, 3, 384, 384]
            return_dict: Ignored, kept for compatibility

        Returns:
            Visual features [batch_size, 144, 4608]
        """
        # Get SigLIP features
        outputs = self.encoder(pixel_values=pixel_values)

        # Get patch embeddings (excluding CLS token if present)
        # SigLIP doesn't use CLS token, so we get all patch embeddings
        hidden_states = outputs.last_hidden_state  # [batch_size, 576, 1152]

        # Apply pixel shuffle
        hidden_states = self.pixel_shuffle(hidden_states)  # [batch_size, 144, 4608]

        return hidden_states

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of encoder parameters."""
        return next(self.encoder.parameters()).dtype

    @property
    def device(self) -> torch.device:
        """Return the device of encoder parameters."""
        return next(self.encoder.parameters()).device


class VisionEncoderConfig:
    """Configuration for vision encoder."""

    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        pixel_shuffle_scale: int = 2,
        freeze: bool = True,
        image_size: int = 384,
    ):
        self.model_name = model_name
        self.pixel_shuffle_scale = pixel_shuffle_scale
        self.freeze = freeze
        self.image_size = image_size

        # Computed values for SigLIP-so400m
        self.hidden_size = 1152
        self.num_patches = 576
        self.num_output_tokens = self.num_patches // (pixel_shuffle_scale ** 2)
        self.output_hidden_size = self.hidden_size * (pixel_shuffle_scale ** 2)
