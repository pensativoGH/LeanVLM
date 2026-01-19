"""Processor for combining image and text processing."""

import torch
from PIL import Image
from transformers import SiglipImageProcessor, AutoTokenizer
from typing import List, Dict, Any, Optional, Union


class VLMProcessor:
    """
    Unified processor for TinyVLM that handles both image and text processing.

    Combines:
    - SigLIP image processor for 512x512 images
    - SmolLM2 tokenizer for text
    """

    def __init__(
        self,
        vision_model_name: str = "google/siglip-base-patch16-384",
        language_model_name: str = "HuggingFaceTB/SmolLM2-135M",
        image_size: int = 384,  # Must match SigLIP's native size
        max_length: int = 512,
    ):
        self.image_size = image_size
        self.max_length = max_length

        # Image processor
        self.image_processor = SiglipImageProcessor.from_pretrained(vision_model_name)
        # Override size to our target
        self.image_processor.size = {"height": image_size, "width": image_size}

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            language_model_name,
            trust_remote_code=True,
        )

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def process_image(self, image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """
        Process image(s) for the vision encoder.

        Args:
            image: PIL Image or list of PIL Images

        Returns:
            Tensor of shape (B, 3, 512, 512)
        """
        if isinstance(image, Image.Image):
            image = [image]

        # Convert to RGB if needed
        images = []
        for img in image:
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)

        # Process with SigLIP processor
        processed = self.image_processor(
            images=images,
            return_tensors="pt",
        )

        return processed["pixel_values"]

    def process_text(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = True,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize text.

        Args:
            text: String or list of strings
            add_special_tokens: Whether to add special tokens
            padding: Whether to pad sequences
            truncation: Whether to truncate long sequences
            max_length: Maximum sequence length

        Returns:
            Dict with input_ids and attention_mask
        """
        if max_length is None:
            max_length = self.max_length

        encoded = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt",
        )

        return encoded

    def format_conversation(
        self,
        user_text: str,
        assistant_text: str,
        use_chat_template: bool = True,
    ) -> str:
        """
        Format a conversation turn into the expected format.

        For instruction-tuned models (SmolLM2-Instruct), uses the proper chat template.
        Falls back to simple format for base models.

        Args:
            user_text: User's question/prompt
            assistant_text: Assistant's response
            use_chat_template: Whether to use tokenizer's chat template (default: True)

        Returns:
            Formatted conversation string
        """
        if use_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                # Format as chat messages
                messages = [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ]
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                # Fall back to simple format if chat template fails
                pass

        # Simple format for base models
        eos = self.tokenizer.eos_token
        return f"User: {user_text}\nAssistant: {assistant_text}{eos}"

    def format_prompt(
        self,
        user_text: str,
        use_chat_template: bool = True,
    ) -> str:
        """
        Format a user prompt for generation (without assistant response).

        Args:
            user_text: User's question/prompt
            use_chat_template: Whether to use tokenizer's chat template

        Returns:
            Formatted prompt string ready for generation
        """
        if use_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                messages = [{"role": "user", "content": user_text}]
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,  # Adds assistant prompt prefix
                )
            except Exception:
                pass

        # Simple format for base models (space after colon to match training format)
        return f"User: {user_text}\nAssistant: "

    def get_assistant_start_position(self, input_ids: torch.Tensor) -> int:
        """
        Find the position where assistant response starts in the tokenized sequence.

        Handles multiple formats:
        - Chat template: <|im_start|>assistant\\n
        - Simple format: Assistant:

        This is used to create labels where we only predict the assistant's response.

        Args:
            input_ids: Tokenized input IDs

        Returns:
            Position index where assistant response starts
        """
        input_ids_list = input_ids.tolist()
        if isinstance(input_ids_list[0], list):
            input_ids_list = input_ids_list[0]

        # Try chat template format first: <|im_start|>assistant\n
        # The assistant content starts AFTER the newline following "assistant"
        chat_markers = [
            "<|im_start|>assistant\n",  # SmolLM2-Instruct format
            "<|im_start|>assistant",     # Without newline
        ]

        for marker in chat_markers:
            marker_tokens = self.tokenizer.encode(marker, add_special_tokens=False)
            for i in range(len(input_ids_list) - len(marker_tokens) + 1):
                if input_ids_list[i:i + len(marker_tokens)] == marker_tokens:
                    return i + len(marker_tokens)

        # Try simple format: "Assistant:"
        assistant_tokens = self.tokenizer.encode("Assistant:", add_special_tokens=False)
        for i in range(len(input_ids_list) - len(assistant_tokens) + 1):
            if input_ids_list[i:i + len(assistant_tokens)] == assistant_tokens:
                return i + len(assistant_tokens)

        # If not found, return -1
        return -1

    def add_special_tokens(self, tokens: List[str]) -> int:
        """
        Add special tokens to the tokenizer.

        Args:
            tokens: List of special token strings to add

        Returns:
            Number of tokens added
        """
        special_tokens = {"additional_special_tokens": tokens}
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        return num_added

    def get_think_end_position(self, input_ids: torch.Tensor, think_end_id: int) -> int:
        """
        Find position after </think> token for answer extraction.

        Args:
            input_ids: Token IDs (1D tensor)
            think_end_id: Token ID for </think>

        Returns:
            Position index after </think> token, or -1 if not found
        """
        input_ids_list = input_ids.tolist()
        if isinstance(input_ids_list[0], list):
            input_ids_list = input_ids_list[0]

        try:
            pos = input_ids_list.index(think_end_id)
            return pos + 1  # Return position AFTER </think>
        except ValueError:
            return -1

    def format_conversation_with_mode(
        self,
        user_text: str,
        assistant_text: str,
        mode: str = "think",
        use_chat_template: bool = True,
    ) -> str:
        """
        Format a conversation with mode tag prepended.

        Args:
            user_text: User's question/prompt
            assistant_text: Assistant's response
            mode: "think" or "no_think"
            use_chat_template: Whether to use tokenizer's chat template

        Returns:
            Formatted conversation string with mode tag
        """
        mode_tag = "/think" if mode == "think" else "/no_think"
        tagged_user_text = f"{mode_tag} {user_text}"
        return self.format_conversation(tagged_user_text, assistant_text, use_chat_template)

    def save_tokenizer(self, path: str) -> None:
        """Save tokenizer to disk."""
        self.tokenizer.save_pretrained(path)

    def load_tokenizer(self, path: str) -> None:
        """Load tokenizer from disk (updates internal tokenizer)."""
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)
