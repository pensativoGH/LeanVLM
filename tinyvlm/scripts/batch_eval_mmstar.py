"""Batch evaluate multiple checkpoints on MMStar and save results."""

import argparse
import json
import os
import sys
import csv
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.model import TinyVLM
from src.data import VLMProcessor
from src.evaluation.mmstar import MMStarEvaluator, MMStarDataset


def evaluate_checkpoint(checkpoint_path: str, data_dir: str, device: str = "cuda") -> dict:
    """Evaluate a single checkpoint and return results."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {checkpoint_path}")
    print(f"{'='*60}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    # Get model config
    model_config = config.get("model", {})
    vision_encoder = model_config.get("vision_encoder", "google/siglip-base-patch16-384")
    language_model = model_config.get("language_model", "HuggingFaceTB/SmolLM2-360M-Instruct")
    image_size = model_config.get("image_size", 384)
    num_image_tokens = model_config.get("num_image_tokens", 144)

    # Create model
    model = TinyVLM(
        vision_encoder_name=vision_encoder,
        language_model_name=language_model,
        image_size=image_size,
        num_image_tokens=num_image_tokens,
        freeze_vision=True,
        freeze_lm=True,
    )

    # Load weights
    model.vision_encoder.load_state_dict(checkpoint["model_state_dict"]["vision_encoder"])
    model.projector.load_state_dict(checkpoint["model_state_dict"]["projector"])
    model.language_model.model.load_state_dict(checkpoint["model_state_dict"]["language_model"])

    model.to(device)
    model.to(torch.bfloat16)
    model.eval()

    # Create processor
    processor = VLMProcessor(
        vision_model_name=vision_encoder,
        language_model_name=language_model,
        image_size=image_size,
    )

    # Check if instruct model
    use_chat_template = "instruct" in language_model.lower()

    # Create dataset
    dataset = MMStarDataset(
        data_dir=data_dir,
        processor=processor,
    )

    # Create evaluator
    evaluator = MMStarEvaluator(
        model=model,
        processor=processor,
        device=device,
        use_chat_template=use_chat_template,
    )

    # Run evaluation
    summary = evaluator.evaluate(dataset=dataset, verbose=True)

    # Convert to dict format
    results = {
        "overall_accuracy": summary.accuracy * 100,
        "correct": summary.correct,
        "total": summary.total,
        "per_category": {},
    }

    for cat, data in summary.per_category.items():
        results["per_category"][cat] = {
            "accuracy": data["accuracy"] * 100,
            "correct": data["correct"],
            "total": data["total"],
        }

    # Clean up to free memory
    del model
    del processor
    del evaluator
    del dataset
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate checkpoints on MMStar")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints_v2",
        help="Directory containing checkpoints",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/mmstar",
        help="Path to MMStar data",
    )
    parser.add_argument(
        "--step-interval",
        type=int,
        default=4000,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/mmstar/batch_eval_results.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=2,
        help="Training stage to evaluate (1 or 2)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Find checkpoints
    checkpoint_dir = Path(args.checkpoint_dir)
    pattern = f"stage{args.stage}_step*.pt"
    checkpoints = sorted(checkpoint_dir.glob(pattern))

    # Filter by step interval
    eval_checkpoints = []
    for ckpt in checkpoints:
        # Extract step number
        name = ckpt.stem  # e.g., "stage2_step4000"
        step = int(name.split("step")[1])
        if step % args.step_interval == 0:
            eval_checkpoints.append((step, ckpt))

    eval_checkpoints.sort(key=lambda x: x[0])

    print(f"\nFound {len(eval_checkpoints)} checkpoints to evaluate:")
    for step, ckpt in eval_checkpoints:
        print(f"  Step {step}: {ckpt.name}")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run evaluations and collect results
    all_results = []

    for step, ckpt in eval_checkpoints:
        try:
            results = evaluate_checkpoint(str(ckpt), args.data_dir, device)

            row = {
                "step": step,
                "checkpoint": ckpt.name,
                "overall_accuracy": results["overall_accuracy"],
                "correct": results["correct"],
                "total": results["total"],
            }

            # Add per-category results
            for category, data in results["per_category"].items():
                row[f"{category}_accuracy"] = data["accuracy"]
                row[f"{category}_correct"] = data["correct"]
                row[f"{category}_total"] = data["total"]

            all_results.append(row)

            # Save intermediate results
            with open(output_path, "w", newline="") as f:
                if all_results:
                    writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                    writer.writeheader()
                    writer.writerows(all_results)

            print(f"\nStep {step}: {results['overall_accuracy']:.2f}%")

        except Exception as e:
            print(f"Error evaluating {ckpt}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final summary
    print(f"\n{'='*60}")
    print("BATCH EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_path}")

    print("\nSummary:")
    print("-" * 80)
    print(f"{'Step':>8} | {'Overall':>8} | {'CP':>6} | {'FP':>6} | {'IR':>6} | {'LR':>6} | {'Math':>6} | {'ST':>6}")
    print("-" * 80)
    for row in all_results:
        print(f"{row['step']:>8} | {row['overall_accuracy']:>7.2f}% | "
              f"{row.get('CP_accuracy', 0):>5.1f}% | {row.get('FP_accuracy', 0):>5.1f}% | "
              f"{row.get('IR_accuracy', 0):>5.1f}% | {row.get('LR_accuracy', 0):>5.1f}% | "
              f"{row.get('Math_accuracy', 0):>5.1f}% | {row.get('ST_accuracy', 0):>5.1f}%")
    print("-" * 80)

    # Also save as JSON for easier programmatic access
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"JSON results saved to: {json_path}")


if __name__ == "__main__":
    main()
