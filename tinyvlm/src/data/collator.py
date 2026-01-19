"""Collator for batching VLM samples."""

import torch
from typing import List, Dict, Any

from .processor import VLMProcessor


class VLMCollator:
    """
    Collator for batching VLM samples.

    Handles:
    - Padding text sequences to the same length
    - Stacking images
    - Creating attention masks for the full sequence (image + text)
    - Creating labels with -100 for image tokens and user text
    """

    def __init__(
        self,
        processor: VLMProcessor,
        num_image_tokens: int = 144,  # 12x12 after pixel shuffle
    ):
        """
        Initialize the collator.

        Args:
            processor: VLMProcessor with tokenizer
            num_image_tokens: Number of image tokens (144 after pixel shuffle)
        """
        self.processor = processor
        self.num_image_tokens = num_image_tokens
        self.pad_token_id = processor.pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.

        Args:
            batch: List of samples, each with pixel_values, input_ids, attention_mask

        Returns:
            Batched tensors:
            - pixel_values: (B, 3, 384, 384)
            - input_ids: (B, max_text_len)
            - attention_mask: (B, 144 + max_text_len)
            - labels: (B, 144 + max_text_len)
        """
        # Filter out None samples (from corrupt/problematic images)
        batch = [sample for sample in batch if sample is not None]

        if len(batch) == 0:
            # Return empty batch - dataloader will skip
            raise ValueError("All samples in batch were None/invalid")

        # Stack pixel values
        pixel_values = torch.stack([sample["pixel_values"] for sample in batch])

        # Get max text length
        text_lengths = [sample["input_ids"].shape[0] for sample in batch]
        max_text_len = max(text_lengths)

        batch_size = len(batch)

        # Initialize padded tensors
        input_ids = torch.full(
            (batch_size, max_text_len),
            self.pad_token_id,
            dtype=torch.long,
        )
        text_attention_mask = torch.zeros(
            (batch_size, max_text_len),
            dtype=torch.long,
        )

        # Fill in each sample
        for i, sample in enumerate(batch):
            seq_len = sample["input_ids"].shape[0]
            input_ids[i, :seq_len] = sample["input_ids"]
            text_attention_mask[i, :seq_len] = sample["attention_mask"]

        # Create full attention mask (image tokens + text tokens)
        # Image tokens always have attention = 1
        image_attention = torch.ones(
            (batch_size, self.num_image_tokens),
            dtype=torch.long,
        )
        full_attention_mask = torch.cat([image_attention, text_attention_mask], dim=1)

        # Create labels
        # -100 for image tokens (don't predict)
        # -100 for user text (only predict assistant response)
        # actual token IDs for assistant response
        labels = self._create_labels(input_ids, batch)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": full_attention_mask,
            "labels": labels,
        }

    def _create_labels(
        self,
        input_ids: torch.Tensor,
        batch: List[Dict[str, Any]],
    ) -> torch.Tensor:
        """
        Create labels for training.

        Labels are:
        - -100 for image tokens (num_image_tokens positions)
        - -100 for everything before and including "Assistant:"
        - Actual token IDs for the assistant's response
        - -100 for padding

        Args:
            input_ids: (B, max_text_len) padded input IDs
            batch: Original batch samples

        Returns:
            (B, num_image_tokens + max_text_len) labels tensor
        """
        batch_size, max_text_len = input_ids.shape
        total_len = self.num_image_tokens + max_text_len

        # Initialize all labels as -100
        labels = torch.full(
            (batch_size, total_len),
            -100,
            dtype=torch.long,
        )

        # For each sample, find where assistant response starts
        for i, sample in enumerate(batch):
            # Get the original unpadded input_ids
            orig_input_ids = sample["input_ids"]

            # Find "Assistant:" position
            assistant_start = self.processor.get_assistant_start_position(orig_input_ids)

            if assistant_start > 0:
                # Copy labels starting from assistant response
                # Position in full sequence = num_image_tokens + position_in_text
                text_len = orig_input_ids.shape[0]

                for j in range(assistant_start, text_len):
                    full_pos = self.num_image_tokens + j
                    labels[i, full_pos] = orig_input_ids[j]

        return labels
