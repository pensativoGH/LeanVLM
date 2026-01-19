"""
Curriculum sampler for training on progressively longer sequences.

This sampler orders training data by token length, starting with shorter
sequences and progressively including longer ones. This helps the model
learn the thinking format on simpler examples first.
"""

import torch
from torch.utils.data import Sampler
from typing import List, Iterator, Optional
import numpy as np


class CurriculumSampler(Sampler[int]):
    """
    Sampler that orders data by sequence length for curriculum learning.

    The training is divided into stages, each including samples up to
    a certain length threshold. Samples are shuffled within each stage.

    Example stages:
    - Stage A (30%): sequences 0-2000 tokens
    - Stage B (30%): sequences 0-4000 tokens (includes Stage A)
    - Stage C (40%): sequences 0-8000 tokens (includes all)

    This allows the model to:
    1. Learn the basic format on shorter, simpler examples
    2. Gradually handle more complex reasoning chains
    3. Avoid destabilization from very long sequences early in training
    """

    def __init__(
        self,
        token_lengths: List[int],
        stages: List[dict] = None,
        total_samples: int = None,
        seed: int = 42,
    ):
        """
        Initialize curriculum sampler.

        Args:
            token_lengths: List of token lengths for each sample in dataset
            stages: List of stage configs, each with:
                    - max_tokens: Maximum token length for this stage
                    - ratio: Proportion of training to spend in this stage
                    Default: [
                        {"max_tokens": 2000, "ratio": 0.30},
                        {"max_tokens": 4000, "ratio": 0.30},
                        {"max_tokens": 8000, "ratio": 0.40},
                    ]
            total_samples: Total number of samples to yield (for steps-based training)
            seed: Random seed for shuffling within stages
        """
        self.token_lengths = np.array(token_lengths)
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Default curriculum stages
        if stages is None:
            stages = [
                {"max_tokens": 2000, "ratio": 0.30},
                {"max_tokens": 4000, "ratio": 0.30},
                {"max_tokens": 8000, "ratio": 0.40},
            ]
        self.stages = stages

        # Determine total samples
        self.total_samples = total_samples or len(token_lengths)

        # Pre-compute indices for each stage
        self._build_stage_indices()

    def _build_stage_indices(self):
        """Build sorted indices for each curriculum stage."""
        self.stage_indices = []

        for stage in self.stages:
            max_tokens = stage["max_tokens"]
            # Get indices of samples that fit in this stage
            valid_mask = self.token_lengths <= max_tokens
            indices = np.where(valid_mask)[0]
            # Sort by length within stage
            sorted_indices = indices[np.argsort(self.token_lengths[indices])]
            self.stage_indices.append(sorted_indices)

    def __iter__(self) -> Iterator[int]:
        """
        Iterate through samples in curriculum order.

        Samples are yielded stage by stage, with shuffling within each stage.
        """
        samples_yielded = 0

        for stage_idx, stage in enumerate(self.stages):
            # Calculate samples for this stage
            stage_samples = int(self.total_samples * stage["ratio"])

            # Handle last stage - yield remaining
            if stage_idx == len(self.stages) - 1:
                stage_samples = self.total_samples - samples_yielded

            if stage_samples <= 0:
                continue

            # Get indices for this stage
            indices = self.stage_indices[stage_idx].copy()

            # Shuffle within stage
            self.rng.shuffle(indices)

            # Yield samples (with cycling if needed)
            yielded_in_stage = 0
            idx_pos = 0

            while yielded_in_stage < stage_samples and samples_yielded < self.total_samples:
                yield indices[idx_pos % len(indices)]
                yielded_in_stage += 1
                samples_yielded += 1
                idx_pos += 1

    def __len__(self) -> int:
        return self.total_samples


class LengthSortedSampler(Sampler[int]):
    """
    Simple sampler that sorts samples by length.

    This provides a basic curriculum where samples are processed
    from shortest to longest, without explicit stages.
    """

    def __init__(
        self,
        token_lengths: List[int],
        ascending: bool = True,
        shuffle_within_buckets: bool = True,
        bucket_size: int = 1000,
        seed: int = 42,
    ):
        """
        Initialize length-sorted sampler.

        Args:
            token_lengths: List of token lengths for each sample
            ascending: If True, shortest first; if False, longest first
            shuffle_within_buckets: Shuffle samples within length buckets
            bucket_size: Size of buckets for shuffling
            seed: Random seed
        """
        self.token_lengths = np.array(token_lengths)
        self.ascending = ascending
        self.shuffle_within_buckets = shuffle_within_buckets
        self.bucket_size = bucket_size
        self.rng = np.random.RandomState(seed)

        # Sort indices by length
        sorted_indices = np.argsort(self.token_lengths)
        if not ascending:
            sorted_indices = sorted_indices[::-1]

        # Optionally shuffle within buckets
        if shuffle_within_buckets:
            self.indices = self._shuffle_within_buckets(sorted_indices)
        else:
            self.indices = sorted_indices

    def _shuffle_within_buckets(self, sorted_indices: np.ndarray) -> np.ndarray:
        """Shuffle samples within length buckets to add variety."""
        result = []
        for i in range(0, len(sorted_indices), self.bucket_size):
            bucket = sorted_indices[i:i + self.bucket_size].copy()
            self.rng.shuffle(bucket)
            result.extend(bucket)
        return np.array(result)

    def __iter__(self) -> Iterator[int]:
        for idx in self.indices:
            yield idx

    def __len__(self) -> int:
        return len(self.indices)


class MixedCurriculumSampler(Sampler[int]):
    """
    Sampler that mixes curriculum learning with mode-balanced sampling.

    For dual-mode training, this ensures:
    - Curriculum ordering (short to long) applies to thinking mode samples
    - Non-thinking samples are distributed evenly across training
    """

    def __init__(
        self,
        token_lengths: List[int],
        modes: List[str],
        stages: List[dict] = None,
        total_samples: int = None,
        seed: int = 42,
    ):
        """
        Initialize mixed curriculum sampler.

        Args:
            token_lengths: List of token lengths for each sample
            modes: List of modes ("think" or "no_think") for each sample
            stages: Curriculum stages (only applies to thinking samples)
            total_samples: Total samples to yield
            seed: Random seed
        """
        self.token_lengths = np.array(token_lengths)
        self.modes = np.array(modes)
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Default stages
        if stages is None:
            stages = [
                {"max_tokens": 2000, "ratio": 0.30},
                {"max_tokens": 4000, "ratio": 0.30},
                {"max_tokens": 8000, "ratio": 0.40},
            ]
        self.stages = stages

        # Separate thinking and non-thinking indices
        self.think_indices = np.where(self.modes == "think")[0]
        self.no_think_indices = np.where(self.modes == "no_think")[0]

        self.total_samples = total_samples or len(token_lengths)

        # Build curriculum for thinking samples
        self._build_think_curriculum()

    def _build_think_curriculum(self):
        """Build curriculum ordered indices for thinking samples."""
        think_lengths = self.token_lengths[self.think_indices]

        self.think_stage_indices = []
        for stage in self.stages:
            max_tokens = stage["max_tokens"]
            valid_mask = think_lengths <= max_tokens
            valid_in_think = np.where(valid_mask)[0]
            # Map back to original dataset indices
            original_indices = self.think_indices[valid_in_think]
            # Sort by length
            sorted_order = np.argsort(self.token_lengths[original_indices])
            self.think_stage_indices.append(original_indices[sorted_order])

    def __iter__(self) -> Iterator[int]:
        """
        Yield samples with curriculum for thinking and even distribution for non-thinking.
        """
        # Shuffle non-thinking indices
        no_think_shuffled = self.no_think_indices.copy()
        self.rng.shuffle(no_think_shuffled)

        # Calculate how many non-thinking samples per batch of samples
        think_count = len(self.think_indices)
        no_think_count = len(self.no_think_indices)
        total = think_count + no_think_count

        no_think_per_sample = no_think_count / total if total > 0 else 0

        samples_yielded = 0
        no_think_yielded = 0
        no_think_idx = 0

        for stage_idx, stage in enumerate(self.stages):
            stage_samples = int(self.total_samples * stage["ratio"])

            if stage_idx == len(self.stages) - 1:
                stage_samples = self.total_samples - samples_yielded

            # Get thinking indices for this stage
            think_indices = self.think_stage_indices[stage_idx].copy()
            self.rng.shuffle(think_indices)

            think_idx = 0

            for _ in range(stage_samples):
                if samples_yielded >= self.total_samples:
                    break

                # Decide whether to yield thinking or non-thinking
                expected_no_think = int(samples_yielded * no_think_per_sample)
                should_yield_no_think = (
                    no_think_yielded < expected_no_think and
                    no_think_idx < len(no_think_shuffled)
                )

                if should_yield_no_think:
                    yield no_think_shuffled[no_think_idx]
                    no_think_idx = (no_think_idx + 1) % len(no_think_shuffled)
                    no_think_yielded += 1
                else:
                    yield think_indices[think_idx % len(think_indices)]
                    think_idx += 1

                samples_yielded += 1

    def __len__(self) -> int:
        return self.total_samples


def create_curriculum_sampler(
    dataset,
    stages: List[dict] = None,
    total_samples: int = None,
    seed: int = 42,
) -> CurriculumSampler:
    """
    Create a curriculum sampler from a dataset.

    The dataset must have a `token_lengths` attribute or `get_token_length` method.

    Args:
        dataset: Dataset with token length information
        stages: Curriculum stages
        total_samples: Total samples to yield
        seed: Random seed

    Returns:
        CurriculumSampler instance
    """
    # Get token lengths from dataset
    if hasattr(dataset, "token_lengths") and dataset.token_lengths is not None:
        token_lengths = dataset.token_lengths
    elif hasattr(dataset, "get_token_length"):
        token_lengths = [dataset.get_token_length(i) for i in range(len(dataset))]
    else:
        raise ValueError("Dataset must have token_lengths attribute or get_token_length method")

    # Check for modes (dual-mode training)
    if hasattr(dataset, "modes"):
        return MixedCurriculumSampler(
            token_lengths=token_lengths,
            modes=dataset.modes,
            stages=stages,
            total_samples=total_samples,
            seed=seed,
        )
    else:
        return CurriculumSampler(
            token_lengths=token_lengths,
            stages=stages,
            total_samples=total_samples,
            seed=seed,
        )
