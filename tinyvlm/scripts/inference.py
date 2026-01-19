#!/usr/bin/env python3
"""Inference script for TinyVLM."""

import argparse
import torch
from pathlib import Path
from PIL import Image
import sys
import os

# Add project root to path (handles running from any directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)  # Ensure relative paths work

from src.model import TinyVLM
from src.data import VLMProcessor


def load_model(checkpoint_path: str, config: dict) -> TinyVLM:
    """Load a trained TinyVLM model from checkpoint."""
    model = TinyVLM(
        vision_encoder_name=config["model"]["vision_encoder"],
        language_model_name=config["model"]["language_model"],
        image_size=config["model"]["image_size"],
        num_image_tokens=config["model"]["num_image_tokens"],
        freeze_vision=True,
        freeze_lm=True,  # Freeze for inference
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.vision_encoder.load_state_dict(
        checkpoint["model_state_dict"]["vision_encoder"]
    )
    model.projector.load_state_dict(
        checkpoint["model_state_dict"]["projector"]
    )
    model.language_model.model.load_state_dict(
        checkpoint["model_state_dict"]["language_model"]
    )

    return model


def generate_response(
    model: TinyVLM,
    processor: VLMProcessor,
    image_path: str,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """Generate a response for an image and prompt using chat template."""
    # Load and process image
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor.process_image(image).to(device)

    # Format prompt using chat template (SmolLM2-Instruct format)
    formatted_prompt = processor.format_prompt(prompt, use_chat_template=True)

    # Tokenize prompt
    encoded = processor.process_text(
        formatted_prompt,
        add_special_tokens=True,
        padding=False,
        truncation=True,
    )
    input_ids = encoded["input_ids"].to(device)

    # Create attention mask for full sequence (image + text)
    num_image_tokens = model.num_image_tokens
    text_len = input_ids.shape[1]
    attention_mask = torch.ones(
        (1, num_image_tokens + text_len),
        dtype=torch.long,
        device=device,
    )

    # Convert to bf16
    pixel_values = pixel_values.to(torch.bfloat16)

    # Generate
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=temperature > 0,
            pad_token_id=processor.pad_token_id,
            eos_token_id=processor.eos_token_id,
        )

    # Decode the generated tokens
    # Note: When using inputs_embeds, output may be shorter than input
    input_length = input_ids.shape[1]
    output_length = output_ids.shape[1]

    if output_length > input_length:
        # Output includes input tokens - extract only generated part
        generated_ids = output_ids[0, input_length:]
    else:
        # Output contains only generated tokens
        generated_ids = output_ids[0]

    response = processor.tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    return response


def interactive_mode(
    model: TinyVLM,
    processor: VLMProcessor,
    device: torch.device,
):
    """Run interactive inference mode."""
    print("\n" + "=" * 50)
    print("TinyVLM Interactive Mode")
    print("=" * 50)
    print("Commands:")
    print("  /image <path>  - Load a new image")
    print("  /quit          - Exit")
    print("=" * 50 + "\n")

    current_image = None

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            print("Goodbye!")
            break

        if user_input.lower().startswith("/image "):
            image_path = user_input[7:].strip()
            if Path(image_path).exists():
                current_image = image_path
                print(f"Loaded image: {image_path}")
            else:
                print(f"Error: Image not found: {image_path}")
            continue

        if current_image is None:
            print("Please load an image first with: /image <path>")
            continue

        # Generate response
        try:
            response = generate_response(
                model=model,
                processor=processor,
                image_path=current_image,
                prompt=user_input,
                device=device,
            )
            print(f"Assistant: {response}\n")
        except Exception as e:
            print(f"Error generating response: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="TinyVLM Inference")
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
        help="Path to image file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt/question about the image",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    args = parser.parse_args()

    # Load checkpoint to get config
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})

    # Use defaults if config not in checkpoint
    if not config:
        config = {
            "model": {
                "vision_encoder": "google/siglip-base-patch16-384",
                "language_model": "HuggingFaceTB/SmolLM2-360M-Instruct",
                "image_size": 384,
                "num_image_tokens": 144,
            }
        }

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create processor
    processor = VLMProcessor(
        vision_model_name=config["model"]["vision_encoder"],
        language_model_name=config["model"]["language_model"],
        image_size=config["model"]["image_size"],
    )

    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, config)
    model.to(device)
    model.to(torch.bfloat16)
    print("Model loaded!")

    if args.interactive:
        interactive_mode(model, processor, device)
    elif args.image and args.prompt:
        response = generate_response(
            model=model,
            processor=processor,
            image_path=args.image,
            prompt=args.prompt,
            device=device,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"\nResponse: {response}")
    else:
        parser.print_help()
        print("\nError: Provide --interactive OR both --image and --prompt")


if __name__ == "__main__":
    main()
