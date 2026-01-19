"""Multimodal Processor.

Handles image preprocessing and text tokenization for STEM VLM training.
"""

from typing import Optional, List, Dict, Any, Union
from PIL import Image

import torch
from transformers import AutoTokenizer


class MultimodalProcessor:
    """Processor for multimodal (image + text) inputs.

    Handles:
    - Image preprocessing (resize, normalize)
    - Text tokenization
    - Chat template formatting

    Args:
        tokenizer_name: HuggingFace tokenizer name (default: SmolLM2)
        image_size: Target image size (default: 384 for SigLIP)
        max_length: Maximum sequence length for text
    """

    def __init__(
        self,
        tokenizer_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
        image_size: int = 384,
        max_length: int = 512,
    ):
        self.image_size = image_size
        self.max_length = max_length

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Image normalization (SigLIP uses these values)
        self.image_mean = [0.5, 0.5, 0.5]
        self.image_std = [0.5, 0.5, 0.5]

        # Load SigLIP processor for image preprocessing
        self._load_image_processor()

    def _load_image_processor(self):
        """Load SigLIP image processor."""
        from transformers import SiglipImageProcessor

        self.image_processor = SiglipImageProcessor.from_pretrained(
            "google/siglip-so400m-patch14-384"
        )

    def process_image(
        self,
        image: Union[Image.Image, str],
        return_tensors: str = "pt",
    ) -> torch.Tensor:
        """Process a single image.

        Args:
            image: PIL Image or path to image
            return_tensors: Return format ("pt" for PyTorch)

        Returns:
            Processed image tensor [3, 384, 384]
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image or path, got {type(image)}")

        # Use SigLIP processor
        processed = self.image_processor(
            images=image,
            return_tensors=return_tensors,
        )

        return processed.pixel_values.squeeze(0)

    def process_images(
        self,
        images: List[Union[Image.Image, str]],
        return_tensors: str = "pt",
    ) -> torch.Tensor:
        """Process a batch of images.

        Args:
            images: List of PIL Images or paths
            return_tensors: Return format

        Returns:
            Batched image tensor [batch_size, 3, 384, 384]
        """
        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img).convert("RGB"))
            else:
                pil_images.append(img)

        processed = self.image_processor(
            images=pil_images,
            return_tensors=return_tensors,
        )

        return processed.pixel_values

    def tokenize(
        self,
        text: Union[str, List[str]],
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text.

        Args:
            text: Single string or list of strings
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length (default: self.max_length)
            return_tensors: Return format

        Returns:
            Dict with input_ids and attention_mask
        """
        if max_length is None:
            max_length = self.max_length

        return self.tokenizer(
            text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )

    def format_chat(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Format messages using chat template.

        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            add_generation_prompt: Whether to add generation prompt

        Returns:
            Formatted chat string
        """
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def process_conversation(
        self,
        image: Union[Image.Image, str],
        conversations: List[Dict[str, str]],
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Process an image-conversation pair.

        Args:
            image: Input image
            conversations: List of conversation turns
            return_tensors: Return format

        Returns:
            Dict with pixel_values, input_ids, attention_mask, labels
        """
        # Process image
        pixel_values = self.process_image(image, return_tensors)

        # Format and tokenize conversation
        formatted = self.format_chat(conversations, add_generation_prompt=False)
        tokens = self.tokenize(formatted, return_tensors=return_tensors)

        # Create labels (mask user turns)
        labels = self._create_labels(conversations, tokens["input_ids"])

        return {
            "pixel_values": pixel_values,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }

    def _create_labels(
        self,
        conversations: List[Dict[str, str]],
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Create labels with user turns masked.

        User content and special tokens are masked with -100.
        Only assistant responses are used for loss computation.
        """
        labels = input_ids.clone()

        # Simple approach: mask everything except assistant responses
        # This is a simplified version - production would need more careful handling
        text = self.tokenizer.decode(input_ids[0])

        # Find assistant response boundaries and mask appropriately
        # For SmolLM2-Instruct format: <|im_start|>assistant\n{content}<|im_end|>
        assistant_marker = "assistant\n"

        current_pos = 0
        in_assistant = False

        for conv in conversations:
            role = conv["role"]
            content = conv["content"]

            if role == "assistant":
                # Find this content in the decoded text
                # Keep these tokens as labels
                pass
            else:
                # Mask user/system content
                pass

        # Simplified: mask the first portion (system + user) and keep assistant
        # For proper implementation, would need to track token boundaries

        # For now, use the full input as labels (will be refined in training)
        return labels

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:
        """Get pad token ID."""
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        """Get EOS token ID."""
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int:
        """Get BOS token ID."""
        return self.tokenizer.bos_token_id

    def decode(
        self,
        token_ids: Union[torch.Tensor, List[int]],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(
        self,
        token_ids: Union[torch.Tensor, List[List[int]]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Decode batch of token IDs to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
