"""Training Utilities.

Helper functions for STEM VLM training including parameter grouping,
learning rate scheduling, and logging.
"""

import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Iterator

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler


def setup_logging(
    log_dir: Optional[str] = None,
    log_level: int = logging.INFO,
    log_to_file: bool = True,
) -> logging.Logger:
    """Setup logging configuration.

    Args:
        log_dir: Directory for log files
        log_level: Logging level
        log_to_file: Whether to log to file

    Returns:
        Logger instance
    """
    logger = logging.getLogger("stem_vlm")
    logger.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file and log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(log_dir) / f"train_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)

    return logger


def get_stem_parameter_groups(
    model: nn.Module,
    lr_stem_embeddings: float = 1e-3,
    lr_projector: float = 5e-5,
    lr_default: float = 5e-5,
    weight_decay: float = 0.01,
) -> List[Dict[str, Any]]:
    """Create parameter groups with different learning rates for STEM training.

    STEM training benefits from:
    - Higher LR for STEM embeddings (1e-3) - need to learn from scratch
    - Lower LR for pretrained weights (5e-5) - fine-tuning

    Args:
        model: STEM VLM model
        lr_stem_embeddings: Learning rate for STEM embeddings
        lr_projector: Learning rate for projector
        lr_default: Default learning rate for other parameters
        weight_decay: Weight decay coefficient

    Returns:
        List of parameter group dicts for optimizer
    """
    # Separate parameters by type
    stem_embedding_params = []
    projector_params = []
    decay_params = []
    no_decay_params = []

    # No decay for biases, layernorm, embeddings
    no_decay_keywords = ["bias", "layernorm", "layer_norm", "ln_", "embed_tokens"]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # STEM embeddings
        if "stem_embeddings" in name:
            stem_embedding_params.append(param)
        # Projector
        elif "projector" in name:
            projector_params.append(param)
        # No decay parameters
        elif any(nd in name.lower() for nd in no_decay_keywords):
            no_decay_params.append(param)
        # Regular decay parameters
        else:
            decay_params.append(param)

    parameter_groups = [
        {
            "params": stem_embedding_params,
            "lr": lr_stem_embeddings,
            "weight_decay": 0.0,  # No decay for embeddings
            "name": "stem_embeddings",
        },
        {
            "params": projector_params,
            "lr": lr_projector,
            "weight_decay": weight_decay,
            "name": "projector",
        },
        {
            "params": decay_params,
            "lr": lr_default,
            "weight_decay": weight_decay,
            "name": "decay",
        },
        {
            "params": no_decay_params,
            "lr": lr_default,
            "weight_decay": 0.0,
            "name": "no_decay",
        },
    ]

    # Filter out empty groups
    parameter_groups = [g for g in parameter_groups if len(g["params"]) > 0]

    return parameter_groups


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
    last_epoch: int = -1,
) -> LambdaLR:
    """Create cosine learning rate schedule with linear warmup.

    Args:
        optimizer: Optimizer instance
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        min_lr_ratio: Minimum LR as ratio of initial LR
        last_epoch: The index of last epoch (for resuming)

    Returns:
        LR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def create_optimizer(
    model: nn.Module,
    lr_stem_embeddings: float = 1e-3,
    lr_projector: float = 5e-5,
    lr_default: float = 5e-5,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    use_8bit: bool = False,
) -> torch.optim.Optimizer:
    """Create optimizer with STEM-specific parameter groups.

    Args:
        model: STEM VLM model
        lr_stem_embeddings: LR for STEM embeddings
        lr_projector: LR for projector
        lr_default: Default LR
        weight_decay: Weight decay
        betas: Adam betas
        eps: Adam epsilon
        use_8bit: Use 8-bit Adam from bitsandbytes (saves ~27GB memory)

    Returns:
        Configured optimizer
    """
    parameter_groups = get_stem_parameter_groups(
        model,
        lr_stem_embeddings=lr_stem_embeddings,
        lr_projector=lr_projector,
        lr_default=lr_default,
        weight_decay=weight_decay,
    )

    if use_8bit:
        try:
            import bitsandbytes as bnb
            logging.getLogger("stem_vlm").info("Using 8-bit Adam optimizer (saves ~27GB memory)")
            return bnb.optim.Adam8bit(parameter_groups, betas=betas, eps=eps)
        except ImportError:
            logging.getLogger("stem_vlm").warning(
                "bitsandbytes not installed, falling back to AdamW. "
                "Install with: pip install bitsandbytes"
            )
            return AdamW(parameter_groups, betas=betas, eps=eps)

    return AdamW(parameter_groups, betas=betas, eps=eps)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters by component.

    Args:
        model: Model to count

    Returns:
        Dict with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count by component
    counts = {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }

    # STEM-specific counts
    stem_count = 0
    for name, param in model.named_parameters():
        if "stem_embeddings" in name:
            stem_count += param.numel()
    counts["stem_embeddings"] = stem_count

    return counts


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    epoch: int,
    step: int,
    loss: float,
    save_path: str,
    config: Optional[Dict[str, Any]] = None,
    scaler: Optional[GradScaler] = None,
):
    """Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: LR scheduler state
        epoch: Current epoch
        step: Current step
        loss: Current loss
        save_path: Path to save checkpoint
        config: Model/training config
        scaler: Optional GradScaler for fp16 training
    """
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    scaler: Optional[GradScaler] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state
        scaler: Optional GradScaler to load state
        device: Device to load tensors to (default: cpu)

    Returns:
        Checkpoint metadata (epoch, step, loss, config)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu", weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "loss": checkpoint.get("loss", float("inf")),
        "config": checkpoint.get("config"),
    }


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_grad_norm(model: nn.Module) -> float:
    """Compute gradient norm across all parameters.

    Args:
        model: Model with gradients

    Returns:
        Total gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def seed_everything(seed: int):
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
