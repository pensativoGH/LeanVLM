#!/usr/bin/env python3
"""MMStar benchmark evaluation script for TinyVLM.

Evaluates a trained TinyVLM checkpoint on the MMStar benchmark,
computing overall accuracy and per-category breakdown.

Supports both SmolLM2-135M (base) and SmolLM2-360M-Instruct models.

Usage:
    python scripts/eval_mmstar.py --checkpoint checkpoints/stage2_final.pt
    python scripts/eval_mmstar.py --checkpoint checkpoints/stage2_final.pt --num-samples 100
    python scripts/eval_mmstar.py --checkpoint checkpoints/stage2_final.pt --debug
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import TinyVLM
from src.data import VLMProcessor
from src.evaluation.mmstar import MMStarDataset, MMStarEvaluator, save_results


def is_instruct_model(model_name: str) -> bool:
    """Check if the model is an instruct-tuned variant."""
    model_lower = model_name.lower()
    return "instruct" in model_lower or "chat" in model_lower


def load_model(checkpoint_path: str, device: torch.device, tokenizer_path: str = None) -> tuple:
    """
    Load TinyVLM model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: torch device
        tokenizer_path: Path to custom tokenizer (for post-training checkpoints with extended vocab)

    Returns:
        Tuple of (model, config, tokenizer_path if extended vocab)
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = checkpoint.get("config", {})
    if not config or "model" not in config:
        # Default config (fallback for old checkpoints without config)
        print("Warning: Checkpoint missing config, using defaults")
        config = {
            "model": {
                "vision_encoder": "google/siglip-base-patch16-384",
                "language_model": "HuggingFaceTB/SmolLM2-135M",
                "image_size": 384,
                "num_image_tokens": 144,
            }
        }

    # Create model
    model = TinyVLM(
        vision_encoder_name=config["model"]["vision_encoder"],
        language_model_name=config["model"]["language_model"],
        image_size=config["model"]["image_size"],
        num_image_tokens=config["model"]["num_image_tokens"],
        freeze_vision=True,
        freeze_lm=True,
    )

    # Check if we need to resize embeddings for extended vocabulary
    # Post-training checkpoints have <think> and </think> tokens
    lm_state_dict = checkpoint["model_state_dict"]["language_model"]
    checkpoint_vocab_size = lm_state_dict["model.embed_tokens.weight"].shape[0]
    model_vocab_size = model.language_model.vocab_size

    if checkpoint_vocab_size != model_vocab_size:
        print(f"Resizing embeddings: {model_vocab_size} -> {checkpoint_vocab_size}")
        model.resize_token_embeddings(checkpoint_vocab_size)

    # Load weights
    model.vision_encoder.load_state_dict(
        checkpoint["model_state_dict"]["vision_encoder"]
    )
    model.projector.load_state_dict(
        checkpoint["model_state_dict"]["projector"]
    )
    model.language_model.model.load_state_dict(
        checkpoint["model_state_dict"]["language_model"]
    )

    model.to(device)
    model.to(torch.bfloat16)
    model.eval()

    return model, config, checkpoint_vocab_size


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TinyVLM on MMStar benchmark",
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
        default="data/mmstar",
        help="Path to local MMStar data (default: data/mmstar)",
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
        default=5,
        help="Max tokens to generate per answer (default: 5, sufficient for A/B/C/D)",
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
    parser.add_argument(
        "--use-chat-template",
        type=str,
        default="auto",
        choices=["auto", "yes", "no"],
        help="Use chat template for prompts (auto detects from model name)",
    )
    parser.add_argument(
        "--prompt-style",
        type=str,
        default="raw",
        choices=["raw", "instruct"],
        help="Prompt style: 'raw' (nanoVLM/lmms-eval style, just question) or 'instruct' (adds answer instruction)",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to custom tokenizer (for post-training checkpoints with extended vocab)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["think", "no_think", None],
        help="Prefix prompts with /think or /no_think for thinking-trained models",
    )
    args = parser.parse_args()

    # Setup device (CUDA > MPS > CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    model, config, vocab_size = load_model(args.checkpoint, device, args.tokenizer_path)
    lm_name = config["model"]["language_model"]
    print(f"Model loaded: {lm_name}")
    print(f"Vocabulary size: {vocab_size}")

    # Determine whether to use chat template
    if args.use_chat_template == "auto":
        use_chat_template = is_instruct_model(lm_name)
    else:
        use_chat_template = args.use_chat_template == "yes"

    print(f"Using chat template: {use_chat_template}")
    print(f"Prompt style: {args.prompt_style}")

    # Create processor
    processor = VLMProcessor(
        vision_model_name=config["model"]["vision_encoder"],
        language_model_name=config["model"]["language_model"],
        image_size=config["model"]["image_size"],
    )

    # Load custom tokenizer if provided (for post-training checkpoints)
    if args.tokenizer_path:
        print(f"Loading custom tokenizer from: {args.tokenizer_path}")
        processor.load_tokenizer(args.tokenizer_path)

    # Load dataset
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Local data not found at {data_dir}")
        print("Will stream from HuggingFace (slower)")
        data_dir = None

    dataset = MMStarDataset(
        data_dir=str(data_dir) if data_dir else None,
        processor=processor,
    )

    # Create evaluator
    evaluator = MMStarEvaluator(
        model=model,
        processor=processor,
        device=device,
        max_new_tokens=args.max_new_tokens,
        use_chat_template=use_chat_template,
        debug=args.debug,
        prompt_style=args.prompt_style,
        mode=args.mode,
    )

    if args.mode:
        print(f"Mode: /{args.mode}")

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
    output_file = output_dir / "eval_results.json"
    save_results(summary, output_file, include_details=args.save_details)

    print("\n" + "=" * 50)
    print(f"Results saved to: {output_file}")
    print("=" * 50)

    return summary


if __name__ == "__main__":
    main()
