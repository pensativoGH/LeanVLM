#!/usr/bin/env python3
"""MMStar benchmark evaluation script for MM-STEM.

Evaluates a trained STEM VLM checkpoint on the MMStar benchmark,
computing overall accuracy and per-category breakdown.

Usage:
    python scripts/eval_mmstar.py --checkpoint outputs/stem_vlm_smollm2/checkpoints/step_24000.pt
    python scripts/eval_mmstar.py --checkpoint outputs/stem_vlm_smollm2/checkpoints/step_24000.pt --num-samples 100
    python scripts/eval_mmstar.py --checkpoint outputs/stem_vlm_smollm2/checkpoints/step_24000.pt --debug
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.stem_vlm import STEMVLM, STEMVLMConfig
from src.data.processor import MultimodalProcessor
from src.evaluation.mmstar import MMStarDataset, MMStarEvaluator, save_results


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """
    Load STEM VLM model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: torch device

    Returns:
        Tuple of (model, config)
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Get config from checkpoint
    config_dict = checkpoint.get("config", None)

    if config_dict:
        print(f"Found config in checkpoint")
        # Reconstruct config
        config = STEMVLMConfig(
            vision_model_name=config_dict.get("vision_model_name", "google/siglip-base-patch16-384"),
            pixel_shuffle_scale=config_dict.get("pixel_shuffle_scale", 2),
            freeze_vision_encoder=config_dict.get("freeze_vision_encoder", True),
            image_size=config_dict.get("image_size", 384),
            projector_type=config_dict.get("projector_type", "mlp"),
            projector_num_layers=config_dict.get("projector_num_layers", 2),
            projector_dropout=config_dict.get("projector_dropout", 0.0),
            lm_model_name=config_dict.get("lm_model_name", "HuggingFaceTB/SmolLM2-360M-Instruct"),
            vocab_size=config_dict.get("vocab_size", 49152),
            hidden_size=config_dict.get("hidden_size", 960),
            intermediate_size=config_dict.get("intermediate_size", 2560),
            num_hidden_layers=config_dict.get("num_hidden_layers", 32),
            num_attention_heads=config_dict.get("num_attention_heads", 15),
            num_key_value_heads=config_dict.get("num_key_value_heads", 5),
            max_position_embeddings=config_dict.get("max_position_embeddings", 8192),
            rope_theta=config_dict.get("rope_theta", 100000.0),
            rms_norm_eps=config_dict.get("rms_norm_eps", 1e-5),
            hidden_act=config_dict.get("hidden_act", "silu"),
            tie_word_embeddings=config_dict.get("tie_word_embeddings", True),
            stem_init_std=config_dict.get("stem_init_std", 0.02),
        )
    else:
        print("Warning: No config in checkpoint, using defaults")
        config = STEMVLMConfig()

    # Create model
    print("Creating model...")
    model = STEMVLM(config)

    # Load weights
    print("Loading weights...")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Move to device and set dtype
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    # Enable STEM for evaluation
    model.enable_stem()

    print(f"Model loaded successfully")
    print(f"  Vision: {config.vision_model_name}")
    print(f"  LM: {config.lm_model_name}")
    print(f"  Image tokens: {model.num_image_tokens}")

    return model, config


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MM-STEM on MMStar benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to local MMStar data (default: download from HF)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/mmstar",
        help="Directory to save results (default: results/mmstar)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit number of samples (for testing)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Max tokens to generate per answer (default: 32)",
    )
    parser.add_argument(
        "--save-details",
        action="store_true",
        help="Save per-sample predictions to results",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info for first few samples",
    )
    args = parser.parse_args()

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    model, config = load_model(args.checkpoint, device)

    # Create processor
    processor = MultimodalProcessor(
        tokenizer_name=config.lm_model_name,
        image_size=config.image_size,
        max_length=2048,
    )

    # Load dataset
    dataset = MMStarDataset(
        data_dir=args.data_dir,
        processor=processor,
    )

    # Create evaluator
    evaluator = MMStarEvaluator(
        model=model,
        processor=processor,
        device=device,
        max_new_tokens=args.max_new_tokens,
        debug=args.debug,
    )

    # Run evaluation
    print("\n" + "=" * 50)
    print("Starting MMStar Evaluation")
    print("=" * 50)

    summary = evaluator.evaluate(
        dataset=dataset,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        verbose=True,
    )

    # Print results
    print("\n" + "=" * 50)
    print("MMStar Evaluation Results")
    print("=" * 50)
    print(f"\nOverall Accuracy: {summary.accuracy * 100:.2f}%")
    print(f"Correct: {summary.correct} / {summary.total}")

    print("\nPer-Category Breakdown:")
    print("-" * 40)
    for cat, stats in sorted(summary.per_category.items()):
        acc = stats["accuracy"] * 100
        print(f"  {cat:6s}: {acc:5.2f}% ({stats['correct']:3d}/{stats['total']:3d})")

    # Save results
    output_dir = Path(args.output_dir)
    checkpoint_name = Path(args.checkpoint).stem
    output_file = output_dir / f"eval_{checkpoint_name}.json"
    save_results(summary, output_file, include_details=args.save_details)

    print("\n" + "=" * 50)
    print(f"Results saved to: {output_file}")
    print("=" * 50)

    return summary


if __name__ == "__main__":
    main()
