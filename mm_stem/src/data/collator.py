"""Multimodal Data Collator.

Handles batching and padding for multimodal training data.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass
class MultimodalCollator:
    """Collator for multimodal (image + text) batches.

    Handles:
    - Stacking images (assuming same size)
    - Padding text sequences
    - Creating attention masks
    - Handling labels with proper ignore index

    Args:
        pad_token_id: Token ID for padding (default: 0)
        label_pad_token_id: Label ID for padding (default: -100)
        max_length: Maximum sequence length (optional truncation)
    """

    pad_token_id: int = 0
    label_pad_token_id: int = -100
    max_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of features.

        Args:
            features: List of dicts with pixel_values, input_ids, attention_mask, labels

        Returns:
            Batched dict with properly padded tensors
        """
        batch = {}

        # Stack images (assumes all same size)
        if "pixel_values" in features[0]:
            pixel_values = [f["pixel_values"] for f in features]
            batch["pixel_values"] = torch.stack(pixel_values, dim=0)

        # Pad text sequences
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]

        # Get max length in batch
        max_len = max(ids.shape[0] for ids in input_ids)
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)

        # Pad input_ids and attention_mask
        batch["input_ids"] = self._pad_sequence(
            input_ids, max_len, self.pad_token_id
        )
        batch["attention_mask"] = self._pad_sequence(
            attention_mask, max_len, 0
        )

        # Pad labels if present
        if "labels" in features[0]:
            labels = [f["labels"] for f in features]
            batch["labels"] = self._pad_sequence(
                labels, max_len, self.label_pad_token_id
            )

        return batch

    def _pad_sequence(
        self,
        sequences: List[torch.Tensor],
        max_len: int,
        pad_value: int,
    ) -> torch.Tensor:
        """Pad sequences to max_len with right padding.

        Args:
            sequences: List of 1D tensors
            max_len: Target length
            pad_value: Value to use for padding

        Returns:
            Padded tensor [batch_size, max_len]
        """
        batch_size = len(sequences)
        padded = torch.full(
            (batch_size, max_len),
            fill_value=pad_value,
            dtype=sequences[0].dtype,
        )

        for i, seq in enumerate(sequences):
            length = min(seq.shape[0], max_len)
            padded[i, :length] = seq[:length]

        return padded


@dataclass
class TextOnlyCollator:
    """Collator for text-only batches.

    Args:
        pad_token_id: Token ID for padding
        label_pad_token_id: Label ID for padding
        max_length: Maximum sequence length
    """

    pad_token_id: int = 0
    label_pad_token_id: int = -100
    max_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate text-only batch."""
        batch = {}

        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]

        max_len = max(ids.shape[0] for ids in input_ids)
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)

        batch["input_ids"] = self._pad_sequence(input_ids, max_len, self.pad_token_id)
        batch["attention_mask"] = self._pad_sequence(attention_mask, max_len, 0)

        if "labels" in features[0]:
            labels = [f["labels"] for f in features]
            batch["labels"] = self._pad_sequence(labels, max_len, self.label_pad_token_id)

        return batch

    def _pad_sequence(
        self,
        sequences: List[torch.Tensor],
        max_len: int,
        pad_value: int,
    ) -> torch.Tensor:
        """Pad sequences with right padding."""
        batch_size = len(sequences)
        padded = torch.full(
            (batch_size, max_len),
            fill_value=pad_value,
            dtype=sequences[0].dtype,
        )

        for i, seq in enumerate(sequences):
            length = min(seq.shape[0], max_len)
            padded[i, :length] = seq[:length]

        return padded


def create_collator(
    processor,
    multimodal: bool = True,
    max_length: Optional[int] = None,
) -> Union[MultimodalCollator, TextOnlyCollator]:
    """Create appropriate collator.

    Args:
        processor: MultimodalProcessor instance
        multimodal: Whether to create multimodal collator
        max_length: Maximum sequence length

    Returns:
        Collator instance
    """
    pad_token_id = processor.pad_token_id

    if multimodal:
        return MultimodalCollator(
            pad_token_id=pad_token_id,
            label_pad_token_id=-100,
            max_length=max_length,
        )
    else:
        return TextOnlyCollator(
            pad_token_id=pad_token_id,
            label_pad_token_id=-100,
            max_length=max_length,
        )
