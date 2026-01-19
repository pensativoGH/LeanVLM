"""Download datasets for TinyVLM training and evaluation.

Supports:
- LLaVA-Pretrain: 558K image-caption pairs for Stage 1 (feature alignment)
- the_cauldron: Curated instruction data for Stage 2 (instruction tuning)
- MMStar: 1,500 sample evaluation benchmark for VLM assessment
"""

import os
import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import json

# Curated high-quality subsets for VLM training
# Selected for diversity and quality, targeting 700K+ samples
# All names verified against the_cauldron available configs
CURATED_SUBSETS = [
    # Visual Question Answering (largest, most diverse)
    "vqav2",               # ~83K, classic VQA
    "aokvqa",              # ~17K, knowledge-based VQA
    # "okvqa",             # DISABLED - broken HF dataset references
    "cocoqa",              # ~46K, COCO-based QA
    "visual7w",            # ~14K, grounded QA
    "textvqa",             # ~22K, text in images
    "tallyqa",             # counting questions
    "ocrvqa",              # OCR-based VQA
    "st_vqa",              # scene text VQA

    # Document/Chart Understanding
    "chartqa",             # ~18K, chart reasoning
    "docvqa",              # ~10K, document QA
    "ai2d",                # ~2K, diagram understanding
    "infographic_vqa",     # ~2K, infographic QA
    "plotqa",              # plot understanding
    "dvqa",                # data visualization QA
    "figureqa",            # figure understanding

    # Science & Reasoning
    "scienceqa",           # ~5K, science reasoning
    "clevr",               # compositional reasoning
    "iconqa",              # icon understanding
    "nlvr2",               # visual reasoning
    "vsr",                 # visual spatial reasoning

    # Reading Comprehension
    "visualmrc",           # visual reading comprehension

    # Captions & Description
    "textcaps",            # ~22K, text-aware captions
    "localized_narratives", # ~200K, detailed image descriptions
]

# Smaller quick-test subset
QUICK_SUBSETS = [
    "ai2d",
    "chartqa",
    "scienceqa",
    "cocoqa",
]


def download_llava_pretrain(output_dir: Path, max_samples: int = None) -> int:
    """
    Download LLaVA-Pretrain dataset for Stage 1 training.

    This dataset contains 558K image-caption pairs from BLIP captions
    of LAION/CC/SBU images. Perfect for projector alignment.

    Args:
        output_dir: Directory to save data
        max_samples: Optional limit on samples

    Returns:
        Number of samples downloaded
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "data"
    metadata_file = output_dir / "metadata.json"

    # Check if already downloaded
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        print(f"LLaVA-Pretrain already downloaded ({metadata['num_samples']} samples)")
        return metadata['num_samples']

    print("Downloading LLaVA-Pretrain dataset...")
    print("This dataset contains 558K image-caption pairs for projector alignment.")

    try:
        # Load from HuggingFace
        # The dataset has two JSON files with different schemas:
        # - blip_laion_cc_sbu_558k.json: has 'conversations' column (what we need)
        # - blip_laion_cc_sbu_558k_meta.json: has 'url', 'blip_caption' columns
        # We need to specify which file to load
        dataset = load_dataset(
            "liuhaotian/LLaVA-Pretrain",
            data_files="blip_laion_cc_sbu_558k.json",
            split="train",
        )

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        # Save to disk
        print(f"Saving {len(dataset)} samples to {data_path}...")
        dataset.save_to_disk(str(data_path))

        # Save metadata
        metadata = {
            "name": "llava-pretrain",
            "num_samples": len(dataset),
            "columns": dataset.column_names,
            "source": "liuhaotian/LLaVA-Pretrain",
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"LLaVA-Pretrain: Downloaded {len(dataset)} samples")
        return len(dataset)

    except Exception as e:
        print(f"LLaVA-Pretrain: FAILED - {e}")
        return 0


def get_subset_info(subset_name: str) -> dict:
    """Get info about a subset without downloading."""
    try:
        dataset = load_dataset(
            "HuggingFaceM4/the_cauldron",
            subset_name,
            split="train",
            streaming=True,
        )
        # Get first item to check structure
        first_item = next(iter(dataset))
        return {
            "name": subset_name,
            "columns": list(first_item.keys()),
            "status": "available",
        }
    except Exception as e:
        return {
            "name": subset_name,
            "status": "error",
            "error": str(e),
        }


def download_subset(
    subset_name: str,
    output_dir: Path,
    max_samples: int = None,
) -> int:
    """
    Download a single subset to disk.

    Args:
        subset_name: Name of the subset
        output_dir: Directory to save data
        max_samples: Optional limit on samples

    Returns:
        Number of samples downloaded
    """
    subset_dir = output_dir / subset_name
    subset_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    metadata_file = subset_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        print(f"  {subset_name}: Already downloaded ({metadata['num_samples']} samples)")
        return metadata['num_samples']

    print(f"  Downloading {subset_name}...")

    try:
        # Load dataset (not streaming - downloads to cache first)
        dataset = load_dataset(
            "HuggingFaceM4/the_cauldron",
            subset_name,
            split="train",
        )

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        # Save to disk in Arrow format (fast loading)
        dataset.save_to_disk(str(subset_dir / "data"))

        # Save metadata
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


def download_mmstar(output_dir: Path) -> int:
    """
    Download MMStar evaluation benchmark.

    MMStar contains 1,500 high-quality samples for evaluating VLMs
    across 6 core capabilities: coarse perception, fine-grained perception,
    instance reasoning, logical reasoning, science & technology, and math.

    Args:
        output_dir: Directory to save data

    Returns:
        Number of samples downloaded
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "data"
    metadata_file = output_dir / "metadata.json"

    # Check if already downloaded
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        print(f"MMStar already downloaded ({metadata['num_samples']} samples)")
        return metadata['num_samples']

    print("Downloading MMStar evaluation benchmark...")
    print("This dataset contains 1,500 samples for VLM evaluation.")

    try:
        # Load from HuggingFace
        dataset = load_dataset("Lin-Chen/MMStar", split="val")

        # Save to disk
        print(f"Saving {len(dataset)} samples to {data_path}...")
        dataset.save_to_disk(str(data_path))

        # Save metadata
        metadata = {
            "name": "mmstar",
            "num_samples": len(dataset),
            "columns": dataset.column_names,
            "source": "Lin-Chen/MMStar",
            "split": "val",
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"MMStar: Downloaded {len(dataset)} samples")
        return len(dataset)

    except Exception as e:
        print(f"MMStar: FAILED - {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for TinyVLM training and evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download LLaVA-Pretrain for Stage 1 (recommended first)
  python scripts/download_data.py --dataset llava-pretrain

  # Download curated cauldron subsets for Stage 2
  python scripts/download_data.py --dataset cauldron

  # Download MMStar evaluation benchmark
  python scripts/download_data.py --dataset mmstar

  # Download all training datasets (not mmstar)
  python scripts/download_data.py --dataset all

  # Quick test with limited samples
  python scripts/download_data.py --dataset llava-pretrain --max-samples 1000
        """
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["llava-pretrain", "cauldron", "mmstar", "all"],
        help="Which dataset to download (default: all, which excludes mmstar)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Base output directory (default: data)",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        default=None,
        help="Specific cauldron subsets to download (default: curated list)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Download only quick-test cauldron subsets (~4 small subsets)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per dataset (for testing)",
    )
    parser.add_argument(
        "--list-subsets",
        action="store_true",
        help="List available curated subsets and exit",
    )
    args = parser.parse_args()

    if args.list_subsets:
        print("Datasets available:")
        print("\n1. LLaVA-Pretrain (Stage 1 - Feature Alignment)")
        print("   - 558K image-caption pairs")
        print("   - Simple BLIP captions for projector training")
        print("\n2. the_cauldron (Stage 2 - Instruction Tuning)")
        print("   Curated subsets:")
        for subset in CURATED_SUBSETS:
            print(f"     - {subset}")
        print(f"\n   Quick subsets (--quick):")
        for subset in QUICK_SUBSETS:
            print(f"     - {subset}")
        print("\n3. MMStar (Evaluation Benchmark)")
        print("   - 1,500 samples for VLM evaluation")
        print("   - 6 core capabilities: CP, FP, IR, LR, ST, Math")
        return

    base_dir = Path(args.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Download LLaVA-Pretrain
    if args.dataset in ["llava-pretrain", "all"]:
        print("=" * 50)
        print("Stage 1 Dataset: LLaVA-Pretrain")
        print("=" * 50)
        llava_dir = base_dir / "llava_pretrain"
        llava_count = download_llava_pretrain(llava_dir, args.max_samples)
        print()

    # Download the_cauldron
    if args.dataset in ["cauldron", "all"]:
        print("=" * 50)
        print("Stage 2 Dataset: the_cauldron")
        print("=" * 50)

        # Determine which subsets to download
        if args.subsets:
            subsets = args.subsets
        elif args.quick:
            subsets = QUICK_SUBSETS
        else:
            subsets = CURATED_SUBSETS

        cauldron_dir = base_dir / "the_cauldron"
        cauldron_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {len(subsets)} subsets to {cauldron_dir}")
        print(f"Subsets: {', '.join(subsets)}")
        if args.max_samples:
            print(f"Max samples per subset: {args.max_samples}")
        print()

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
        print(f"Cauldron download complete!")
        print(f"  Total samples: {total_samples:,}")
        print(f"  Successful: {len(successful)}/{len(subsets)}")
        if failed:
            print(f"  Failed: {', '.join(failed)}")

        # Save overall metadata
        overall_metadata = {
            "subsets": successful,
            "total_samples": total_samples,
            "failed": failed,
        }
        with open(cauldron_dir / "metadata.json", "w") as f:
            json.dump(overall_metadata, f, indent=2)

    # Download MMStar
    if args.dataset == "mmstar":
        print("=" * 50)
        print("Evaluation Benchmark: MMStar")
        print("=" * 50)
        mmstar_dir = base_dir / "mmstar"
        mmstar_count = download_mmstar(mmstar_dir)
        print()

    print()
    print("=" * 50)
    print("All downloads complete!")
    print("=" * 50)
    print(f"Data location: {base_dir.absolute()}")
    print()
    print("To check disk usage:")
    print(f"  du -sh {base_dir}/*")


if __name__ == "__main__":
    main()
