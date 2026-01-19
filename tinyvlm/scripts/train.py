#!/usr/bin/env python3
"""Training script for TinyVLM."""

import argparse
import yaml
from pathlib import Path
import sys
import os

# Add project root to path (handles running from any directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)  # Ensure relative paths work

from src.model import TinyVLM
from src.data import VLMProcessor
from src.training import Trainer


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train TinyVLM")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--stage",
        type=int,
        required=True,
        choices=[1, 2],
        help="Training stage (1: projector only, 2: projector + LM)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    print(f"Training Stage: {args.stage}")

    # Create processor
    processor = VLMProcessor(
        vision_model_name=config["model"]["vision_encoder"],
        language_model_name=config["model"]["language_model"],
        image_size=config["model"]["image_size"],
        max_length=config["data"]["max_length"],
    )
    print("Processor initialized")

    # Determine freeze settings based on stage
    stage_config = config[f"stage{args.stage}"]
    freeze_lm = stage_config["freeze_lm"]
    freeze_vision = stage_config.get("freeze_vision", True)  # Default: frozen

    # Create model
    model = TinyVLM(
        vision_encoder_name=config["model"]["vision_encoder"],
        language_model_name=config["model"]["language_model"],
        image_size=config["model"]["image_size"],
        num_image_tokens=config["model"]["num_image_tokens"],
        freeze_vision=freeze_vision,  # Stage 2: unfrozen for better alignment
        freeze_lm=freeze_lm,
    )
    print("Model initialized")

    # Create trainer
    trainer = Trainer(
        model=model,
        processor=processor,
        config=config,
        stage=args.stage,
    )

    # Start training
    try:
        trainer.train(resume_from=args.resume)
    finally:
        trainer.close()

    print("Training complete!")


if __name__ == "__main__":
    main()
