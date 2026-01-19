"""Vision encoder using SigLIP with pixel shuffle to reduce tokens."""

import math
import torch
import torch.nn as nn
from transformers import SiglipVisionModel, SiglipImageProcessor


class PixelShuffle2D(nn.Module):
    """
    Pixel shuffle for spatial token reduction (like nanoVLM).

    Rearranges spatial patches into channel dimension without information loss.
    For example, 2x2 patches → 1 patch with 4x channels.

    This is superior to adaptive average pooling because it preserves all spatial
    information by reshuffling rather than averaging.
    """

    def __init__(self, downscale_factor: int = 2):
        super().__init__()
        self.r = downscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply pixel shuffle to reduce spatial dimensions.

        Args:
            x: (B, H, W, C) tensor

        Returns:
            (B, H/r, W/r, C*r*r) tensor
        """
        B, H, W, C = x.shape
        r = self.r

        # Reshape: (B, H, W, C) -> (B, H//r, r, W//r, r, C)
        x = x.view(B, H // r, r, W // r, r, C)

        # Permute: (B, H//r, W//r, r, r, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

        # Merge spatial into channels: (B, H//r, W//r, C*r*r)
        x = x.view(B, H // r, W // r, C * r * r)

        return x


class VisionEncoder(nn.Module):
    """
    Vision encoder that wraps SigLIP and applies pixel shuffle.

    Takes 384x384 images, extracts 576 patch embeddings (24x24 grid),
    then applies 2x2 pixel shuffle to get 144 tokens (12x12 grid) with 3072 channels.

    This matches nanoVLM's approach of lossless spatial-to-channel reshuffling.
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-384",
        image_size: int = 384,  # Must match SigLIP's native size
        num_output_tokens: int = 144,  # 12x12 after pixel shuffle
        freeze: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_output_tokens = num_output_tokens

        # Load SigLIP vision model
        self.model = SiglipVisionModel.from_pretrained(model_name)

        # Pixel shuffle: 2x2 spatial → channel merge
        # 24x24 grid (576 tokens) → 12x12 grid (144 tokens)
        self.pixel_shuffle = PixelShuffle2D(downscale_factor=2)

        # Hidden dimension after pixel shuffle: 768 * 4 = 3072
        self.vision_hidden_size = self.model.config.hidden_size  # 768
        self.hidden_size = self.vision_hidden_size * 4  # 3072 after pixel shuffle

        # Freeze if requested
        if freeze:
            self.freeze()

    def freeze(self):
        """Freeze all parameters in the vision encoder."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def unfreeze(self):
        """Unfreeze all parameters in the vision encoder."""
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()

    @property
    def dtype(self):
        """Get the dtype of the model parameters."""
        return next(self.model.parameters()).dtype

    @property
    def device(self):
        """Get the device of the model parameters."""
        return next(self.model.parameters()).device

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through vision encoder with pixel shuffle.

        Args:
            pixel_values: (B, 3, 384, 384) tensor of normalized images

        Returns:
            (B, 144, 3072) tensor of pixel-shuffled patch embeddings
        """
        # Get patch embeddings from SigLIP
        # Input: (B, 3, 384, 384) -> Output: (B, 576, 768) for 24x24 patches
        outputs = self.model(pixel_values=pixel_values)
        hidden_states = outputs.last_hidden_state  # (B, N, D) where N=576

        B, N, D = hidden_states.shape
        H = W = int(math.sqrt(N))  # 24x24 grid for 384x384 images

        # Reshape to spatial grid: (B, 24, 24, 768)
        hidden_states = hidden_states.view(B, H, W, D)

        # Apply pixel shuffle: (B, 24, 24, 768) -> (B, 12, 12, 3072)
        shuffled = self.pixel_shuffle(hidden_states)

        # Reshape to sequence: (B, 144, 3072)
        B, H_new, W_new, D_new = shuffled.shape
        output = shuffled.view(B, H_new * W_new, D_new)

        return output


def get_image_processor(model_name: str = "google/siglip-base-patch16-384", image_size: int = 384):
    """
    Get the image processor for preprocessing images.

    Args:
        model_name: HuggingFace model name
        image_size: Target image size (384x384 for SigLIP)

    Returns:
        SiglipImageProcessor configured for the target size
    """
    processor = SiglipImageProcessor.from_pretrained(model_name)
    # Use SigLIP's native size
    processor.size = {"height": image_size, "width": image_size}
    return processor
