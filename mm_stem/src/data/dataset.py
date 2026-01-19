"""Multimodal Dataset.

Dataset classes for vision-language training with STEM VLM.
Supports various data formats including LLaVA-style JSON.
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Union

import torch
from torch.utils.data import Dataset
from PIL import Image

from .processor import MultimodalProcessor


class MultimodalDataset(Dataset):
    """Dataset for multimodal (image + text) training.

    Supports multiple data formats:
    - LLaVA-style JSON with conversations
    - Simple image-caption pairs
    - Custom format via transform function

    Args:
        data_path: Path to data JSON file or directory
        image_dir: Directory containing images
        processor: MultimodalProcessor instance
        max_length: Maximum text sequence length
        transform: Optional custom transform function
    """

    def __init__(
        self,
        data_path: str,
        image_dir: str,
        processor: Optional[MultimodalProcessor] = None,
        max_length: int = 512,
        transform: Optional[Callable] = None,
    ):
        self.data_path = Path(data_path)
        self.image_dir = Path(image_dir)
        self.max_length = max_length
        self.transform = transform

        # Initialize processor
        if processor is None:
            self.processor = MultimodalProcessor(max_length=max_length)
        else:
            self.processor = processor

        # Load data
        self.data = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from file or directory."""
        if self.data_path.is_file():
            with open(self.data_path, "r") as f:
                data = json.load(f)
        elif self.data_path.is_dir():
            # Load all JSON files in directory
            data = []
            for json_file in self.data_path.glob("*.json"):
                with open(json_file, "r") as f:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        data.extend(file_data)
                    else:
                        data.append(file_data)
        else:
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Returns:
            Dict with pixel_values, input_ids, attention_mask, labels
        """
        item = self.data[idx]

        # Apply custom transform if provided
        if self.transform is not None:
            item = self.transform(item)

        # Get image path
        image_path = self._get_image_path(item)

        # Get conversations
        conversations = self._get_conversations(item)

        # Load and process image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")

        # Process through multimodal processor
        pixel_values = self.processor.process_image(image)

        # Format and tokenize conversation
        formatted = self.processor.format_chat(conversations, add_generation_prompt=False)
        tokens = self.processor.tokenize(
            formatted,
            padding=False,  # Padding done in collator
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Create labels
        labels = self._create_labels(conversations, tokens["input_ids"].squeeze(0))

        return {
            "pixel_values": pixel_values,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": labels,
        }

    def _get_image_path(self, item: Dict[str, Any]) -> Path:
        """Extract image path from data item."""
        # Support multiple formats
        if "image" in item:
            image_name = item["image"]
        elif "image_path" in item:
            image_name = item["image_path"]
        elif "file_name" in item:
            image_name = item["file_name"]
        else:
            raise KeyError(f"No image field found in item: {item.keys()}")

        return self.image_dir / image_name

    def _get_conversations(self, item: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract conversations from data item."""
        # LLaVA format
        if "conversations" in item:
            convs = item["conversations"]
            # Convert LLaVA format to standard format
            result = []
            for conv in convs:
                if "from" in conv and "value" in conv:
                    role = "assistant" if conv["from"] in ["gpt", "assistant"] else "user"
                    result.append({"role": role, "content": conv["value"]})
                elif "role" in conv and "content" in conv:
                    result.append(conv)
            return result

        # Simple caption format
        if "caption" in item:
            return [
                {"role": "user", "content": "Describe this image."},
                {"role": "assistant", "content": item["caption"]},
            ]

        # Question-answer format
        if "question" in item and "answer" in item:
            return [
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["answer"]},
            ]

        raise KeyError(f"Could not extract conversations from item: {item.keys()}")

    def _create_labels(
        self,
        conversations: List[Dict[str, str]],
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Create labels with user turns masked.

        Labels are created by:
        1. Starting with input_ids as labels
        2. Masking user content with -100
        3. Only assistant responses contribute to loss
        """
        labels = input_ids.clone()

        # Decode to find boundaries
        decoded = self.processor.decode(input_ids, skip_special_tokens=False)

        # Mask everything up to and including the first assistant marker
        # This masks system prompt, user content, and special tokens before assistant response

        # Find where assistant response starts
        # SmolLM2-Instruct uses: <|im_start|>assistant\n
        assistant_start_marker = "<|im_start|>assistant\n"
        assistant_end_marker = "<|im_end|>"

        current_pos = 0
        mask_end = 0

        for i, conv in enumerate(conversations):
            role = conv["role"]
            content = conv["content"]

            # Find this turn in the decoded text
            if role == "user":
                # Find and mask user content
                marker = f"<|im_start|>user\n{content}<|im_end|>"
                pos = decoded.find(marker, current_pos)
                if pos >= 0:
                    # Mask this section
                    mask_end = pos + len(marker)
                    current_pos = mask_end
            elif role == "assistant":
                # Don't mask assistant content, but mask the marker
                marker_start = f"<|im_start|>assistant\n"
                pos = decoded.find(marker_start, current_pos)
                if pos >= 0:
                    mask_end = pos + len(marker_start)
                    current_pos = decoded.find(assistant_end_marker, mask_end)
                    if current_pos >= 0:
                        current_pos += len(assistant_end_marker)

        # Convert character positions to token positions (approximate)
        # This is a simplified approach - production would use offset mapping
        prefix_text = decoded[:mask_end] if mask_end > 0 else ""
        prefix_tokens = self.processor.tokenize(
            prefix_text,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )["input_ids"]
        mask_tokens = prefix_tokens.shape[1] if len(prefix_tokens.shape) > 1 else len(prefix_tokens)

        # Mask prefix
        labels[:mask_tokens] = -100

        return labels


class TextOnlyDataset(Dataset):
    """Dataset for text-only training (no images).

    Useful for language model pretraining or fine-tuning.

    Args:
        data_path: Path to text data (JSON or text file)
        processor: MultimodalProcessor instance
        max_length: Maximum sequence length
    """

    def __init__(
        self,
        data_path: str,
        processor: Optional[MultimodalProcessor] = None,
        max_length: int = 512,
    ):
        self.data_path = Path(data_path)
        self.max_length = max_length

        if processor is None:
            self.processor = MultimodalProcessor(max_length=max_length)
        else:
            self.processor = processor

        self.data = self._load_data()

    def _load_data(self) -> List[str]:
        """Load text data."""
        if self.data_path.suffix == ".json":
            with open(self.data_path, "r") as f:
                data = json.load(f)
                # Extract text from various formats
                if isinstance(data, list):
                    if isinstance(data[0], str):
                        return data
                    elif isinstance(data[0], dict):
                        return [item.get("text", str(item)) for item in data]
        else:
            # Plain text file
            with open(self.data_path, "r") as f:
                return f.read().split("\n\n")  # Split by paragraphs

        return []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.data[idx]

        tokens = self.processor.tokenize(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),  # Standard LM training
        }


class CauldronDataset(Dataset):
    """Dataset for the_cauldron HuggingFace format.

    Loads from downloaded the_cauldron subsets with 'images' and 'texts' columns.

    Args:
        data_dir: Directory containing subset folders
        processor: MultimodalProcessor instance
        max_length: Maximum text sequence length
        subsets: List of subset names to load (None = all)
        max_samples: Maximum samples per subset
    """

    def __init__(
        self,
        data_dir: str,
        processor: Optional[MultimodalProcessor] = None,
        max_length: int = 512,
        subsets: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
    ):
        from datasets import load_from_disk, concatenate_datasets

        self.data_dir = Path(data_dir)
        self.max_length = max_length

        if processor is None:
            self.processor = MultimodalProcessor(max_length=max_length)
        else:
            self.processor = processor

        # Load all subsets
        datasets_list = []
        subset_dirs = list(self.data_dir.iterdir())

        for subset_dir in subset_dirs:
            if not subset_dir.is_dir():
                continue
            subset_name = subset_dir.name
            if subsets is not None and subset_name not in subsets:
                continue

            data_path = subset_dir / "data"
            if not data_path.exists():
                continue

            try:
                ds = load_from_disk(str(data_path))
                if max_samples and len(ds) > max_samples:
                    ds = ds.select(range(max_samples))
                datasets_list.append(ds)
                print(f"Loaded {subset_name}: {len(ds)} samples")
            except Exception as e:
                print(f"Failed to load {subset_name}: {e}")

        if datasets_list:
            self.dataset = concatenate_datasets(datasets_list)
            print(f"Total: {len(self.dataset)} samples")
        else:
            raise ValueError(f"No datasets loaded from {data_dir}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]

        # Get image (first image in list)
        images = item.get("images", [])
        if images and len(images) > 0:
            image = images[0]
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")
            else:
                image = image.convert("RGB")
        else:
            # Create blank image if none
            image = Image.new("RGB", (384, 384), color=(128, 128, 128))

        # Get text/conversation
        texts = item.get("texts", [])
        if texts and len(texts) > 0:
            text_item = texts[0]
            if isinstance(text_item, dict):
                user_text = text_item.get("user", "Describe this image.")
                assistant_text = text_item.get("assistant", "")
            else:
                user_text = "Describe this image."
                assistant_text = str(text_item)
        else:
            user_text = "Describe this image."
            assistant_text = "An image."

        conversations = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]

        # Process image
        pixel_values = self.processor.process_image(image)

        # Format and tokenize
        formatted = self.processor.format_chat(conversations, add_generation_prompt=False)
        tokens = self.processor.tokenize(
            formatted,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        # Create labels - mask user content, keep assistant
        labels = input_ids.clone()
        # Simple approach: mask first half as prompt
        mask_len = len(input_ids) // 2
        labels[:mask_len] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class DummyDataset(Dataset):
    """Dummy dataset for testing and debugging.

    Generates random image-text pairs.

    Args:
        num_samples: Number of samples
        image_size: Image dimensions
        max_length: Maximum text length
        vocab_size: Vocabulary size for random tokens
    """

    def __init__(
        self,
        num_samples: int = 100,
        image_size: int = 384,
        max_length: int = 128,
        vocab_size: int = 49152,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.max_length = max_length
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Random image
        pixel_values = torch.randn(3, self.image_size, self.image_size)

        # Random tokens (avoid special tokens at start)
        min_len = min(8, self.max_length - 1)
        seq_len = torch.randint(min_len, self.max_length, (1,)).item()
        input_ids = torch.randint(100, self.vocab_size - 100, (seq_len,))
        attention_mask = torch.ones(seq_len, dtype=torch.long)
        labels = input_ids.clone()

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
