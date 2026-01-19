#!/usr/bin/env python3
"""Evaluation script for STEM VLM.

Usage:
    python scripts/eval_stem.py --checkpoint outputs/stem_vlm/checkpoints/final.pt --image path/to/image.jpg
    python scripts/eval_stem.py --checkpoint outputs/stem_vlm/checkpoints/final.pt --eval-data data/eval.json
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import torch
from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.stem_vlm import STEMVLM, STEMVLMConfig
from src.data.processor import MultimodalProcessor
from src.training.utils import get_device


def load_model(
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> STEMVLM:
    """Load STEM VLM from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Create config from checkpoint
    if "config" in checkpoint and checkpoint["config"]:
        config_dict = checkpoint["config"]
        # Reconstruct model config (simplified - would need full config loading)
        config = STEMVLMConfig()
    else:
        config = STEMVLMConfig()

    # Create model
    model = STEMVLM(config)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device=device, dtype=dtype)
    model.eval()

    print(f"Model loaded successfully")
    return model


def generate_response(
    model: STEMVLM,
    processor: MultimodalProcessor,
    image_path: str,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate response for an image-prompt pair."""
    device = next(model.parameters()).device

    # Load and process image
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor.process_image(image).unsqueeze(0).to(device)

    # Format prompt as chat
    messages = [{"role": "user", "content": prompt}]
    formatted = processor.format_chat(messages, add_generation_prompt=True)

    # Tokenize
    tokens = processor.tokenize(formatted, padding=False, truncation=True)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            eos_token_id=processor.eos_token_id,
        )

    # Decode (skip input tokens)
    generated_text = processor.decode(
        generated_ids[0, input_ids.shape[1]:],
        skip_special_tokens=True,
    )

    return generated_text


def interactive_mode(
    model: STEMVLM,
    processor: MultimodalProcessor,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
):
    """Interactive mode for testing the model."""
    print("\n=== STEM VLM Interactive Mode ===")
    print("Commands:")
    print("  image <path>  - Set the current image")
    print("  prompt <text> - Generate response for prompt")
    print("  quit          - Exit")
    print()

    current_image = None

    while True:
        try:
            user_input = input("> ").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                break

            if user_input.lower().startswith("image "):
                image_path = user_input[6:].strip()
                if os.path.exists(image_path):
                    current_image = image_path
                    print(f"Image set: {image_path}")
                else:
                    print(f"Image not found: {image_path}")
                continue

            if user_input.lower().startswith("prompt "):
                prompt = user_input[7:].strip()
                if current_image is None:
                    print("Please set an image first with: image <path>")
                    continue

                print("Generating response...")
                response = generate_response(
                    model, processor, current_image, prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                print(f"\nResponse: {response}\n")
                continue

            # If no command prefix, treat as prompt
            if current_image is not None:
                response = generate_response(
                    model, processor, current_image, user_input,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                print(f"\nResponse: {response}\n")
            else:
                print("Please set an image first with: image <path>")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("Goodbye!")


def evaluate_on_dataset(
    model: STEMVLM,
    processor: MultimodalProcessor,
    eval_data_path: str,
    image_dir: str,
    output_path: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
) -> List[Dict[str, Any]]:
    """Evaluate model on a dataset."""
    import json

    # Load evaluation data
    with open(eval_data_path, "r") as f:
        eval_data = json.load(f)

    results = []

    for item in tqdm(eval_data, desc="Evaluating"):
        # Get image path
        if "image" in item:
            image_path = os.path.join(image_dir, item["image"])
        elif "image_path" in item:
            image_path = item["image_path"]
        else:
            continue

        # Get prompt
        if "conversations" in item:
            prompt = item["conversations"][0].get("value", "Describe this image.")
        elif "question" in item:
            prompt = item["question"]
        else:
            prompt = "Describe this image."

        # Generate response
        try:
            response = generate_response(
                model, processor, image_path, prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            results.append({
                "image": item.get("image", image_path),
                "prompt": prompt,
                "response": response,
                "ground_truth": item.get("answer", item.get("caption", "")),
            })
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate STEM VLM")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to single image for inference",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image in detail.",
        help="Prompt for single image inference",
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default=None,
        help="Path to evaluation dataset JSON",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/images",
        help="Directory containing images for dataset evaluation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results.json",
        help="Output path for evaluation results",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling probability",
    )
    args = parser.parse_args()

    # Setup
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Create processor
    processor = MultimodalProcessor()

    if args.interactive:
        # Interactive mode
        interactive_mode(
            model, processor,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    elif args.image:
        # Single image inference
        print(f"\nImage: {args.image}")
        print(f"Prompt: {args.prompt}")
        response = generate_response(
            model, processor, args.image, args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"\nResponse: {response}")
    elif args.eval_data:
        # Dataset evaluation
        evaluate_on_dataset(
            model, processor,
            eval_data_path=args.eval_data,
            image_dir=args.image_dir,
            output_path=args.output,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    else:
        print("Please specify --image, --eval-data, or --interactive")
        parser.print_help()


if __name__ == "__main__":
    main()
