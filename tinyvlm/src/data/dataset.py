"""Dataset loader for LLaVA-Pretrain and the_cauldron with local and streaming support."""

import os
import json
import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
from datasets import load_dataset, load_from_disk, interleave_datasets, concatenate_datasets
from typing import Dict, Any, Iterator, Optional, List
from PIL import Image
from pathlib import Path

from .processor import VLMProcessor


# Default data directories
DEFAULT_DATA_DIR = "data/the_cauldron"
LLAVA_PRETRAIN_DIR = "data/llava_pretrain"

# Curated subsets (matches download_data.py)
CURATED_SUBSETS = [
    # Visual Question Answering
    "vqav2",
    "aokvqa",
    "cocoqa",
    "visual7w",
    "textvqa",
    "tallyqa",
    "ocrvqa",
    "st_vqa",
    # Document/Chart Understanding
    "chartqa",
    "docvqa",
    "ai2d",
    "infographic_vqa",
    "plotqa",
    "dvqa",
    "figureqa",
    # Science & Reasoning
    "scienceqa",
    "clevr",
    "iconqa",
    "nlvr2",
    "vsr",
    # Reading Comprehension
    "visualmrc",
    # Captions & Description
    "textcaps",
    "localized_narratives",
]

# Default max samples for training (for diversity with cap)
DEFAULT_MAX_SAMPLES = 800_000

# All subsets for streaming fallback
ALL_SUBSETS = [
    "ai2d", "aokvqa", "chart2text", "chartqa", "clevr", "cocoqa",
    "datikz", "diagram_image_to_text", "docvqa", "dvqa", "figureqa",
    "finqa", "geomverse", "hateful_memes", "hitab", "iam", "iconqa",
    "infographic_vqa", "intergps", "localized_narratives", "mapqa",
    "mimic_cgd", "multihiertt", "nlvr2", "ocrvqa", "plotqa", "raven",
    "rendered_text", "robut_sqa", "robut_wikisql", "robut_wtq",
    "scienceqa", "screen2words", "spot_the_diff", "st_vqa", "tabmwp",
    "tallyqa", "tat_qa", "textcaps", "textvqa", "tqa", "visual7w",
    "visualmrc", "vistext", "vqarad", "vqav2", "vsr", "websight",
]


def get_local_subsets(data_dir: str = DEFAULT_DATA_DIR) -> List[str]:
    """Get list of locally downloaded subsets."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return []

    subsets = []
    for subset_dir in data_path.iterdir():
        if subset_dir.is_dir() and (subset_dir / "metadata.json").exists():
            subsets.append(subset_dir.name)
    return subsets


class LLaVAPretrainDataset(Dataset):
    """
    Dataset for LLaVA-Pretrain (558K image-caption pairs).

    Used for Stage 1 feature alignment training.
    Format: Simple image + BLIP-generated caption pairs.
    """

    def __init__(
        self,
        processor: VLMProcessor,
        data_dir: str = LLAVA_PRETRAIN_DIR,
        seed: int = 42,
    ):
        """
        Initialize LLaVA-Pretrain dataset.

        Args:
            processor: VLMProcessor for image and text processing
            data_dir: Directory containing downloaded dataset
            seed: Random seed for shuffling
        """
        self.processor = processor
        self.data_dir = Path(data_dir)
        self.seed = seed

        # Check if local data exists
        data_path = self.data_dir / "data"
        if not data_path.exists():
            raise RuntimeError(
                f"LLaVA-Pretrain data not found in {data_dir}. "
                "Run: python scripts/download_data.py --dataset llava-pretrain"
            )

        # Load from disk
        print(f"Loading LLaVA-Pretrain from {data_path}...")
        self._dataset = load_from_disk(str(data_path))
        self._dataset = self._dataset.shuffle(seed=seed)
        print(f"Loaded {len(self._dataset)} samples")

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get a single processed sample."""
        sample = self._dataset[idx]
        return self._process_sample(sample)

    def _process_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single LLaVA-Pretrain sample.

        Expected format:
        {
            "image": PIL.Image,
            "conversations": [
                {"from": "human", "value": "<image>\nProvide a brief description."},
                {"from": "gpt", "value": "caption text"}
            ]
        }
        """
        try:
            # Get image
            image = sample.get("image")
            if image is None or not isinstance(image, Image.Image):
                return None

            # Get caption from conversations
            conversations = sample.get("conversations", [])
            if len(conversations) < 2:
                return None

            # Find the assistant/gpt response (caption)
            caption = None
            for turn in conversations:
                if turn.get("from") in ["gpt", "assistant"]:
                    caption = turn.get("value", "")
                    break

            if not caption:
                return None

            # Format as simple caption task
            # For stage 1, we use simple "describe this image" format
            user_text = "Describe this image."
            assistant_text = caption

            # Format conversation
            formatted_text = self.processor.format_conversation(user_text, assistant_text)

            # Process image
            pixel_values = self.processor.process_image(image)

            # Tokenize text
            encoded = self.processor.process_text(
                formatted_text,
                add_special_tokens=True,
                padding=False,
                truncation=True,
            )

            return {
                "pixel_values": pixel_values.squeeze(0),
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "text": formatted_text,
            }

        except Exception as e:
            return None


class LocalCauldronDataset(Dataset):
    """
    Map-style dataset for locally downloaded the_cauldron data.

    Much faster than streaming - supports multi-worker loading.
    Supports max_samples with proportional sampling for diversity.
    """

    def __init__(
        self,
        processor: VLMProcessor,
        data_dir: str = DEFAULT_DATA_DIR,
        subsets: List[str] = None,
        seed: int = 42,
        use_chat_template: bool = True,
        max_samples: int = None,
    ):
        """
        Initialize dataset from local files.

        Args:
            processor: VLMProcessor for image and text processing
            data_dir: Directory containing downloaded subsets
            subsets: List of subsets to use (default: all available)
            seed: Random seed for shuffling
            use_chat_template: Whether to use chat template (True for instruction tuning)
            max_samples: Maximum total samples (proportionally sampled from each subset for diversity)
        """
        self.processor = processor
        self.data_dir = Path(data_dir)
        self.seed = seed
        self.use_chat_template = use_chat_template

        # Find available subsets
        available = get_local_subsets(data_dir)
        if not available:
            raise RuntimeError(f"No local data found in {data_dir}. Run scripts/download_data.py first.")

        if subsets:
            self.subsets = [s for s in subsets if s in available]
        else:
            self.subsets = available

        print(f"Loading {len(self.subsets)} local subsets...")

        # Load all subsets and track sizes
        subset_datasets = []
        subset_sizes = []
        for subset in self.subsets:
            subset_path = self.data_dir / subset / "data"
            if subset_path.exists():
                ds = load_from_disk(str(subset_path))
                subset_datasets.append(ds)
                subset_sizes.append(len(ds))
                print(f"  Loaded {subset}: {len(ds)} samples")

        if not subset_datasets:
            raise RuntimeError("No datasets could be loaded!")

        total_available = sum(subset_sizes)
        print(f"Total available: {total_available:,} samples")

        # Apply max_samples with proportional sampling for diversity
        if max_samples and total_available > max_samples:
            print(f"Capping to {max_samples:,} samples (proportional sampling for diversity)")

            # Calculate proportional samples per subset
            sampled_datasets = []
            actual_total = 0
            for i, (ds, size) in enumerate(zip(subset_datasets, subset_sizes)):
                # Proportional share, but ensure at least some samples from each
                proportion = size / total_available
                target_samples = max(100, int(max_samples * proportion))  # min 100 per subset
                actual_samples = min(target_samples, size)

                # Shuffle and select
                ds_shuffled = ds.shuffle(seed=seed)
                ds_sampled = ds_shuffled.select(range(actual_samples))
                sampled_datasets.append(ds_sampled)
                actual_total += actual_samples
                print(f"    {self.subsets[i]}: {size:,} -> {actual_samples:,} samples")

            subset_datasets = sampled_datasets
            print(f"After proportional sampling: {actual_total:,} samples")

        # Concatenate all datasets
        self._dataset = concatenate_datasets(subset_datasets)

        # Shuffle the combined dataset
        self._dataset = self._dataset.shuffle(seed=seed)

        print(f"Final dataset: {len(self._dataset):,} samples")

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get a single processed sample."""
        sample = self._dataset[idx]
        return self._process_sample(sample)

    def _process_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single sample."""
        try:
            # Get the first image
            images = sample.get("images", [])
            if not images:
                return None

            image = images[0]
            if image is None:
                return None

            # Ensure it's a PIL Image
            if not isinstance(image, Image.Image):
                return None

            # Get the first conversation turn
            texts = sample.get("texts", [])
            if not texts:
                return None

            first_turn = texts[0]
            user_text = first_turn.get("user", "")
            assistant_text = first_turn.get("assistant", "")

            if not user_text or not assistant_text:
                return None

            # Format conversation (use chat template based on setting)
            formatted_text = self.processor.format_conversation(
                user_text, assistant_text, use_chat_template=self.use_chat_template
            )

            # Process image
            pixel_values = self.processor.process_image(image)

            # Tokenize text
            encoded = self.processor.process_text(
                formatted_text,
                add_special_tokens=True,
                padding=False,
                truncation=True,
            )

            return {
                "pixel_values": pixel_values.squeeze(0),
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "text": formatted_text,
            }

        except Exception as e:
            # Return a placeholder that collator will skip
            return None


class StreamingCauldronDataset(IterableDataset):
    """
    Streaming dataset for the_cauldron (fallback when no local data).

    Slower due to network I/O but doesn't require disk space.
    """

    def __init__(
        self,
        processor: VLMProcessor,
        subsets: list = None,
        seed: int = 42,
        use_chat_template: bool = True,
    ):
        self.processor = processor
        self.subsets = subsets or ALL_SUBSETS
        self.seed = seed
        self.use_chat_template = use_chat_template
        self._dataset = None

    def _load_datasets(self):
        """Load and interleave all subsets."""
        if self._dataset is not None:
            return

        from tqdm import tqdm

        print(f"Loading {len(self.subsets)} dataset subsets from HuggingFace (streaming)...")
        print("NOTE: This is slower than local data. Consider running scripts/download_data.py")

        datasets = []
        for subset in tqdm(self.subsets, desc="Loading subsets"):
            try:
                ds = load_dataset(
                    "HuggingFaceM4/the_cauldron",
                    subset,
                    split="train",
                    streaming=True,
                )
                ds = ds.shuffle(seed=self.seed, buffer_size=1000)
                datasets.append(ds)
            except Exception as e:
                tqdm.write(f"Warning: Could not load subset {subset}: {e}")
                continue

        if not datasets:
            raise RuntimeError("No datasets could be loaded!")

        print(f"Loaded {len(datasets)} subsets. Interleaving...")
        self._dataset = interleave_datasets(
            datasets,
            stopping_strategy="all_exhausted",
        )
        print("Dataset ready!")

    def _process_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single sample."""
        try:
            images = sample.get("images", [])
            if not images:
                return None

            image = images[0]
            if image is None or not isinstance(image, Image.Image):
                return None

            texts = sample.get("texts", [])
            if not texts:
                return None

            first_turn = texts[0]
            user_text = first_turn.get("user", "")
            assistant_text = first_turn.get("assistant", "")

            if not user_text or not assistant_text:
                return None

            formatted_text = self.processor.format_conversation(
                user_text, assistant_text, use_chat_template=self.use_chat_template
            )
            pixel_values = self.processor.process_image(image)
            encoded = self.processor.process_text(
                formatted_text,
                add_special_tokens=True,
                padding=False,
                truncation=True,
            )

            return {
                "pixel_values": pixel_values.squeeze(0),
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "text": formatted_text,
            }
        except Exception:
            return None

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self._load_datasets()
        for sample in self._dataset:
            processed = self._process_sample(sample)
            if processed is not None:
                yield processed


def create_dataloader(
    processor: VLMProcessor,
    batch_size: int,
    num_workers: int = 4,
    data_dir: str = DEFAULT_DATA_DIR,
    subsets: list = None,
    seed: int = 42,
    force_streaming: bool = False,
    dataset_type: str = "cauldron",  # "cauldron" or "llava-pretrain"
    num_image_tokens: int = 144,  # Updated for pixel shuffle
    use_chat_template: bool = True,  # True for instruction tuning, False for alignment
    max_samples: int = None,  # Cap total samples (proportional sampling for diversity)
) -> DataLoader:
    """
    Create a DataLoader for training.

    Args:
        processor: VLMProcessor for processing
        batch_size: Batch size
        num_workers: Number of data loading workers (only for local data)
        data_dir: Directory for local data
        subsets: List of subset names (only for cauldron)
        seed: Random seed
        force_streaming: Force streaming mode even if local data exists
        dataset_type: "cauldron" for Stage 2, "llava-pretrain" for Stage 1
        num_image_tokens: Number of image tokens (144 for pixel shuffle)
        use_chat_template: Whether to use chat template format (default: True)
        max_samples: Maximum samples (with proportional sampling across subsets for diversity)

    Returns:
        DataLoader with proper collation
    """
    from .collator import VLMCollator

    if dataset_type == "llava-pretrain":
        # Stage 1: LLaVA-Pretrain (image-caption pairs)
        llava_dir = data_dir if "llava" in data_dir else LLAVA_PRETRAIN_DIR
        print(f"Using LLaVA-Pretrain dataset from {llava_dir}")

        dataset = LLaVAPretrainDataset(
            processor=processor,
            data_dir=llava_dir,
            seed=seed,
        )
        actual_num_workers = num_workers

    else:
        # Stage 2: the_cauldron (instruction data)
        cauldron_dir = data_dir if "cauldron" in data_dir else DEFAULT_DATA_DIR
        local_subsets = get_local_subsets(cauldron_dir)
        use_local = len(local_subsets) > 0 and not force_streaming

        if use_local:
            print(f"Using LOCAL cauldron data from {cauldron_dir}")
            print(f"Available subsets: {', '.join(local_subsets)}")
            print(f"Chat template: {use_chat_template}")
            if max_samples:
                print(f"Max samples: {max_samples:,} (proportional sampling for diversity)")

            dataset = LocalCauldronDataset(
                processor=processor,
                data_dir=cauldron_dir,
                subsets=subsets,
                seed=seed,
                use_chat_template=use_chat_template,
                max_samples=max_samples,
            )
            actual_num_workers = num_workers
        else:
            print("Using STREAMING cauldron data from HuggingFace")
            print("For faster training, run: python scripts/download_data.py")
            print(f"Chat template: {use_chat_template}")

            dataset = StreamingCauldronDataset(
                processor=processor,
                subsets=subsets,
                seed=seed,
                use_chat_template=use_chat_template,
            )
            actual_num_workers = 0

    collator = VLMCollator(
        processor=processor,
        num_image_tokens=num_image_tokens,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=actual_num_workers,
        pin_memory=True,
        shuffle=False,  # Already shuffled in dataset
    )

    return dataloader
