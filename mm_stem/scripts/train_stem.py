#!/usr/bin/env python3
"""Training script for STEM VLM.

Usage:
    python scripts/train_stem.py --config configs/stem_config.yaml
    python scripts/train_stem.py --config configs/stem_config.yaml --resume checkpoints/step_1000.pt
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.stem_vlm import STEMVLM, STEMVLMConfig
from src.data.dataset import MultimodalDataset, DummyDataset, CauldronDataset
from src.data.collator import MultimodalCollator
from src.data.processor import MultimodalProcessor
from src.training.stem_trainer import STEMTrainer, TrainingConfig
from src.training.utils import seed_everything, setup_logging, count_parameters


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_model(config: dict, device: torch.device) -> STEMVLM:
    """Create STEM VLM model from config."""
    model_config = config["model"]

    vlm_config = STEMVLMConfig(
        # Vision
        vision_model_name=model_config["vision"]["model_name"],
        pixel_shuffle_scale=model_config["vision"]["pixel_shuffle_scale"],
        freeze_vision_encoder=model_config["vision"]["freeze"],
        image_size=model_config["vision"]["image_size"],
        # Projector
        projector_type=model_config["projector"]["type"],
        projector_num_layers=model_config["projector"]["num_layers"],
        projector_dropout=model_config["projector"]["dropout"],
        # Language model
        lm_model_name=model_config["language_model"]["model_name"],
        vocab_size=model_config["language_model"]["vocab_size"],
        hidden_size=model_config["language_model"]["hidden_size"],
        intermediate_size=model_config["language_model"]["intermediate_size"],
        num_hidden_layers=model_config["language_model"]["num_hidden_layers"],
        num_attention_heads=model_config["language_model"]["num_attention_heads"],
        num_key_value_heads=model_config["language_model"]["num_key_value_heads"],
        max_position_embeddings=model_config["language_model"]["max_position_embeddings"],
        rope_theta=model_config["language_model"]["rope_theta"],
        rms_norm_eps=model_config["language_model"]["rms_norm_eps"],
        hidden_act=model_config["language_model"]["hidden_act"],
        tie_word_embeddings=model_config["language_model"]["tie_word_embeddings"],
        # STEM
        stem_init_std=model_config["stem"]["init_std"],
    )

    # Create model from pretrained
    print(f"Loading pretrained model from {model_config['language_model']['model_name']}...")
    model = STEMVLM.from_pretrained(
        lm_model_name=model_config["language_model"]["model_name"],
        vision_model_name=model_config["vision"]["model_name"],
        config=vlm_config,
        device=device,
        dtype=torch.bfloat16 if config["training"].get("bf16", False) else torch.float16,
    )

    return model


def create_datasets(config: dict, processor: MultimodalProcessor):
    """Create training and evaluation datasets."""
    data_config = config["data"]

    # Check for cauldron data directory
    cauldron_dir = Path(data_config.get("cauldron_dir", "data/the_cauldron"))

    if cauldron_dir.exists():
        print(f"Loading the_cauldron dataset from {cauldron_dir}...")
        train_dataset = CauldronDataset(
            data_dir=str(cauldron_dir),
            processor=processor,
            max_length=data_config["max_length"],
            max_samples=data_config.get("max_samples_per_subset"),
        )
        print(f"Loaded training dataset: {len(train_dataset)} samples")
    else:
        # Check legacy JSON format
        train_path = Path(data_config.get("train_data", ""))
        image_dir = Path(data_config.get("image_dir", ""))

        if train_path.exists() and image_dir.exists():
            train_dataset = MultimodalDataset(
                data_path=str(train_path),
                image_dir=str(image_dir),
                processor=processor,
                max_length=data_config["max_length"],
            )
            print(f"Loaded training dataset: {len(train_dataset)} samples")
        else:
            print("Data paths not found, using dummy dataset for testing...")
            train_dataset = DummyDataset(
                num_samples=100,
                image_size=384,
                max_length=data_config["max_length"],
            )

    # No separate eval dataset for now
    eval_dataset = None

    return train_dataset, eval_dataset


def create_training_config(config: dict) -> TrainingConfig:
    """Create training configuration from YAML config."""
    train_config = config["training"]

    return TrainingConfig(
        output_dir=train_config["output_dir"],
        experiment_name=train_config["experiment_name"],
        num_epochs=train_config.get("num_epochs", 1),  # Legacy, not used in step-based training
        batch_size=train_config["batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        max_grad_norm=train_config["max_grad_norm"],
        lr_stem_embeddings=train_config["lr_stem_embeddings"],
        lr_projector=train_config["lr_projector"],
        lr_default=train_config["lr_default"],
        weight_decay=train_config["weight_decay"],
        warmup_ratio=train_config["warmup_ratio"],
        min_lr_ratio=train_config["min_lr_ratio"],
        fp16=train_config["fp16"],
        bf16=train_config["bf16"],
        use_8bit_optimizer=train_config.get("use_8bit_optimizer", False),
        stage1_steps=train_config["stage1_steps"],
        stage2_steps=train_config["stage2_steps"],
        save_steps=train_config["save_steps"],
        save_total_limit=train_config["save_total_limit"],
        log_steps=train_config["log_steps"],
        eval_steps=train_config["eval_steps"],
        num_workers=train_config["num_workers"],
        pin_memory=train_config["pin_memory"],
        seed=train_config["seed"],
    )


def load_projector_from_checkpoint(model: STEMVLM, checkpoint_path: str):
    """Load projector weights from TinyVLM Stage 1 checkpoint.

    The checkpoint format should have:
    - model_state_dict['projector']: projector weights
    """
    print(f"Loading projector weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "model_state_dict" in checkpoint and "projector" in checkpoint["model_state_dict"]:
        # TinyVLM checkpoint format
        projector_state = checkpoint["model_state_dict"]["projector"]
    elif "projector" in checkpoint:
        # Direct projector state dict
        projector_state = checkpoint["projector"]
    else:
        raise ValueError(f"Could not find projector weights in checkpoint: {list(checkpoint.keys())}")

    # Map TinyVLM projector keys to MM-STEM projector keys
    # TinyVLM: mlp.0.weight, mlp.0.bias, mlp.2.weight, mlp.2.bias
    # MM-STEM: layers.0.weight, layers.0.bias, layers.2.weight, layers.2.bias
    mapped_state = {}
    for key, value in projector_state.items():
        new_key = key.replace("mlp.", "layers.")
        mapped_state[new_key] = value

    # Check if mapping worked
    model_keys = set(model.projector.state_dict().keys())
    mapped_keys = set(mapped_state.keys())

    if model_keys != mapped_keys:
        print(f"Warning: Key mismatch. Model keys: {model_keys}, Checkpoint keys: {mapped_keys}")
        # Try direct loading without mapping
        if set(projector_state.keys()) == model_keys:
            print("Using direct keys (no mapping needed)")
            mapped_state = projector_state

    model.projector.load_state_dict(mapped_state)
    print(f"Loaded projector weights: {len(mapped_state)} tensors")


def main():
    parser = argparse.ArgumentParser(description="Train STEM VLM")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stem_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--load-projector",
        type=str,
        default=None,
        help="Load projector weights from TinyVLM Stage 1 checkpoint",
    )
    parser.add_argument(
        "--stage2-only",
        action="store_true",
        help="Skip Stage 1, run Stage 2 only (requires --load-projector)",
    )
    parser.add_argument(
        "--dummy-data",
        action="store_true",
        help="Use dummy data for testing",
    )
    args = parser.parse_args()

    # Load configuration
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)

    # Setup
    train_config = create_training_config(config)
    seed_everything(train_config.seed)

    # Setup logging
    logger = setup_logging(
        log_dir=str(Path(train_config.output_dir) / train_config.experiment_name / "logs"),
        log_to_file=True,
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create processor
    processor = MultimodalProcessor(
        tokenizer_name=config["model"]["language_model"]["model_name"],
        image_size=config["model"]["vision"]["image_size"],
        max_length=config["data"]["max_length"],
    )

    # Create model
    print("Creating STEM VLM model...")
    model = create_model(config, device)

    # Log parameter counts
    param_counts = model.get_num_params(trainable_only=False)
    print("\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")

    # Create datasets
    if args.dummy_data:
        train_dataset = DummyDataset(
            num_samples=100,
            image_size=384,
            max_length=config["data"]["max_length"],
        )
        eval_dataset = None
    else:
        train_dataset, eval_dataset = create_datasets(config, processor)

    # Create collator
    collator = MultimodalCollator(
        pad_token_id=processor.pad_token_id,
        label_pad_token_id=-100,
        max_length=config["data"]["max_length"],
    )

    # Create data loaders
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
        collate_fn=collator,
    )

    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=train_config.num_workers,
            pin_memory=train_config.pin_memory,
            collate_fn=collator,
        )

    # Resume from checkpoint if specified
    if args.resume:
        train_config.resume_from_checkpoint = args.resume

    # Create trainer and start training
    print("\nStarting training...")
    trainer = STEMTrainer(
        model=model,
        config=train_config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )

    trainer.train()

    print("\nTraining complete!")
    print(f"Model saved to: {train_config.output_dir}/{train_config.experiment_name}")


if __name__ == "__main__":
    main()
