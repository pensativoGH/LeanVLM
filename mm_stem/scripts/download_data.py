#!/usr/bin/env python3
"""Download datasets for MM-STEM training.

Datasets:
- Stage 1: LLaVA-Pretrain (558K image-caption pairs for projector alignment)
- Stage 2: the_cauldron (curated instruction data for full model training)
"""

import os
import argparse
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Curated high-quality subsets from the_cauldron
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

QUICK_SUBSETS = ["ai2d", "chartqa", "scienceqa", "cocoqa"]


def download_llava_pretrain(output_dir: Path, max_samples: int = None) -> int:
    """Download LLaVA-Pretrain dataset for Stage 1."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "data"
    metadata_file = output_dir / "metadata.json"

    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        print(f"LLaVA-Pretrain already downloaded ({metadata['num_samples']} samples)")
        return metadata['num_samples']

    print("Downloading LLaVA-Pretrain dataset...")
    print("This dataset contains 558K image-caption pairs for projector alignment.")

    try:
        dataset = load_dataset(
            "liuhaotian/LLaVA-Pretrain",
            data_files="blip_laion_cc_sbu_558k.json",
            split="train",
        )

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        print(f"Saving {len(dataset)} samples to {data_path}...")
        dataset.save_to_disk(str(data_path))

        metadata = {
            "name": "llava-pretrain",
            "num_samples": len(dataset),
            "columns": dataset.column_names,
            "source": "liuhaotian/LLaVA-Pretrain",
            "stage": 1,
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"LLaVA-Pretrain: Downloaded {len(dataset)} samples")
        return len(dataset)

    except Exception as e:
        print(f"LLaVA-Pretrain: FAILED - {e}")
        return 0


def download_subset(subset_name: str, output_dir: Path, max_samples: int = None) -> int:
    """Download a single cauldron subset."""
    subset_dir = output_dir / subset_name
    subset_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = subset_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        print(f"  {subset_name}: Already downloaded ({metadata['num_samples']} samples)")
        return metadata['num_samples']

    print(f"  Downloading {subset_name}...")

    try:
        dataset = load_dataset(
            "HuggingFaceM4/the_cauldron",
            subset_name,
            split="train",
        )

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        dataset.save_to_disk(str(subset_dir / "data"))

        metadata = {
            "name": subset_name,
            "num_samples": len(dataset),
            "columns": dataset.column_names,
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  {subset_name}: Downloaded {len(dataset)} samples")
        return len(dataset)

    except Exception as e:
        print(f"  {subset_name}: FAILED - {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Download datasets for MM-STEM training")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["llava-pretrain", "cauldron", "all"],
        help="Which dataset to download",
    )
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Download only quick-test subsets")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per dataset")
    parser.add_argument("--subsets", type=str, nargs="+", default=None, help="Specific cauldron subsets")
    args = parser.parse_args()

    base_dir = Path(args.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: LLaVA-Pretrain
    if args.dataset in ["llava-pretrain", "all"]:
        print("=" * 50)
        print("Stage 1 Dataset: LLaVA-Pretrain")
        print("=" * 50)
        llava_dir = base_dir / "llava_pretrain"
        download_llava_pretrain(llava_dir, args.max_samples)
        print()

    # Stage 2: the_cauldron
    if args.dataset in ["cauldron", "all"]:
        print("=" * 50)
        print("Stage 2 Dataset: the_cauldron")
        print("=" * 50)

        if args.subsets:
            subsets = args.subsets
        elif args.quick:
            subsets = QUICK_SUBSETS
        else:
            subsets = CURATED_SUBSETS

        cauldron_dir = base_dir / "the_cauldron"
        cauldron_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {len(subsets)} subsets...")
        total_samples = 0
        successful = []
        failed = []

        for subset in subsets:
            count = download_subset(subset, cauldron_dir, args.max_samples)
            if count > 0:
                total_samples += count
                successful.append(subset)
            else:
                failed.append(subset)

        print()
        print(f"Cauldron complete! Total: {total_samples:,} samples")
        if failed:
            print(f"Failed: {', '.join(failed)}")

        metadata = {"subsets": successful, "total_samples": total_samples, "failed": failed}
        with open(cauldron_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    print()
    print(f"Data location: {base_dir.absolute()}")


if __name__ == "__main__":
    main()
