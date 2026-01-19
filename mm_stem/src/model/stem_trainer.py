"""STEM Trainer.

Training loop for STEM VLM with support for:
- Two-stage training (projector-only → full model)
- Mixed precision training
- Gradient accumulation
- Checkpointing and logging
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .utils import (
    create_optimizer,
    get_cosine_schedule_with_warmup,
    save_checkpoint,
    load_checkpoint,
    AverageMeter,
    get_grad_norm,
    setup_logging,
    count_parameters,
    get_device,
)


@dataclass
class TrainingConfig:
    """Configuration for STEM VLM training."""

    # Output
    output_dir: str = "outputs"
    experiment_name: str = "stem_vlm"

    # Training
    num_epochs: int = 3
    max_steps: Optional[int] = None
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    # Learning rates (STEM-specific)
    lr_stem_embeddings: float = 1e-3
    lr_projector: float = 5e-5
    lr_default: float = 5e-5
    weight_decay: float = 0.01

    # LR schedule
    warmup_steps: int = 100
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.1

    # Mixed precision
    fp16: bool = True
    bf16: bool = False

    # Memory optimization
    use_8bit_optimizer: bool = False  # Use 8-bit Adam to save ~27GB memory

    # Two-stage training
    stage1_epochs: int = 1  # Projector-only training
    stage2_epochs: int = 2  # Full model training

    # Checkpointing
    save_steps: int = 500
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None

    # Logging
    log_steps: int = 10
    eval_steps: int = 500

    # Hardware
    num_workers: int = 4
    pin_memory: bool = True

    # Seed
    seed: int = 42


class STEMTrainer:
    """Trainer for STEM Vision-Language Model.

    Supports two-stage training:
    1. Stage 1: Train projector only (vision encoder and LM frozen)
    2. Stage 2: Train full model (projector + STEM embeddings + LM)

    Args:
        model: STEM VLM model
        config: Training configuration
        train_dataloader: Training data loader
        eval_dataloader: Optional evaluation data loader
        callbacks: Optional list of callback functions
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        callbacks: Optional[List[Callable]] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.callbacks = callbacks or []

        # Setup device
        self.device = get_device()
        self.model = self.model.to(self.device)

        # Setup output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = setup_logging(
            log_dir=str(self.output_dir / "logs"),
            log_to_file=True,
        )

        # Setup mixed precision
        self.use_amp = config.fp16 or config.bf16
        self.amp_dtype = torch.bfloat16 if config.bf16 else torch.float16
        self.scaler = GradScaler() if config.fp16 else None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        # Initialize optimizer and scheduler (done in train())
        self.optimizer = None
        self.scheduler = None

    def train(self):
        """Run full training loop."""
        self.logger.info("Starting STEM VLM training")
        self.logger.info(f"Output directory: {self.output_dir}")

        # Log parameter counts
        param_counts = count_parameters(self.model)
        self.logger.info(f"Model parameters: {param_counts}")

        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            self._load_checkpoint(self.config.resume_from_checkpoint)

        # Two-stage training
        total_epochs = self.config.stage1_epochs + self.config.stage2_epochs

        for epoch in range(self.epoch, total_epochs):
            self.epoch = epoch

            # Determine training stage
            if epoch < self.config.stage1_epochs:
                self._setup_stage1()
                stage_name = "Stage 1 (Projector)"
            else:
                self._setup_stage2()
                stage_name = "Stage 2 (Full Model)"

            self.logger.info(f"Epoch {epoch + 1}/{total_epochs} - {stage_name}")

            # Train one epoch
            train_loss = self._train_epoch()

            # Evaluate
            if self.eval_dataloader is not None:
                eval_loss = self._evaluate()
                self.logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")

                if eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    self._save_checkpoint("best")
            else:
                self.logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")

            # Save epoch checkpoint
            self._save_checkpoint(f"epoch_{epoch + 1}")

            # Early stopping based on max_steps
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                self.logger.info(f"Reached max_steps ({self.config.max_steps}), stopping training")
                break

        # Save final model
        self._save_checkpoint("final")
        self.logger.info("Training complete!")

    def _setup_stage1(self):
        """Setup for Stage 1: projector + first LM layer training (no STEM).

        In Stage 1, we:
        - Train projector to align vision and language
        - Train first LM layer to adapt to image tokens
        - Disable STEM embeddings (use dense computation instead)
          because STEM embeddings are randomly initialized
        """
        # Freeze vision encoder
        self.model.freeze_vision_encoder()

        # Disable STEM - use dense computation for all tokens
        # STEM embeddings are random, so we don't want to use them yet
        self.model.disable_stem()
        self.logger.info("Stage 1: STEM disabled (using dense computation)")

        # Freeze entire language model first
        for name, param in self.model.language_model.named_parameters():
            param.requires_grad = False

        # Unfreeze first layer of LM (layer 0)
        # This includes attention and MLP of layer 0
        for name, param in self.model.language_model.named_parameters():
            if "layers.0." in name:
                param.requires_grad = True
                self.logger.info(f"Stage 1: Unfreezing {name}")

        # Unfreeze projector
        for param in self.model.projector.parameters():
            param.requires_grad = True

        # Create optimizer for projector + first LM layer
        self.optimizer = create_optimizer(
            self.model,
            lr_stem_embeddings=self.config.lr_stem_embeddings,  # For layer 0 STEM
            lr_projector=self.config.lr_projector,
            lr_default=self.config.lr_default,  # For layer 0 attention/other
            weight_decay=self.config.weight_decay,
            use_8bit=self.config.use_8bit_optimizer,
        )

        # Calculate total steps for this stage
        steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
        total_steps = steps_per_epoch * self.config.stage1_epochs

        # Create scheduler
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=self.config.min_lr_ratio,
        )

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Stage 1: {trainable:,} trainable parameters (projector only)")

    def _setup_stage2(self):
        """Setup for Stage 2: full model training with STEM enabled.

        In Stage 2, we:
        - Enable STEM embeddings (now warmed up projector/layer 0 helps)
        - Train full language model including STEM embeddings
        - Keep vision encoder frozen
        """
        # Keep vision encoder frozen
        self.model.freeze_vision_encoder()

        # Enable STEM - now use embedding lookup for text tokens
        self.model.enable_stem()
        self.logger.info("Stage 2: STEM enabled (using embedding lookup for text)")

        # Unfreeze language model
        for param in self.model.language_model.parameters():
            param.requires_grad = True

        # Unfreeze projector
        for param in self.model.projector.parameters():
            param.requires_grad = True

        # Create optimizer with STEM-specific learning rates
        self.optimizer = create_optimizer(
            self.model,
            lr_stem_embeddings=self.config.lr_stem_embeddings,
            lr_projector=self.config.lr_projector,
            lr_default=self.config.lr_default,
            weight_decay=self.config.weight_decay,
            use_8bit=self.config.use_8bit_optimizer,
        )

        # Calculate total steps for this stage
        steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
        total_steps = steps_per_epoch * self.config.stage2_epochs

        # Create scheduler
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=self.config.min_lr_ratio,
        )

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Stage 2: {trainable:,} trainable parameters (full model)")

    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        loss_meter = AverageMeter()

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch + 1}",
            disable=False,
        )

        self.optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update metrics
            loss_meter.update(loss.item() * self.config.gradient_accumulation_steps)

            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                # Logging
                if self.global_step % self.config.log_steps == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    grad_norm = get_grad_norm(self.model)
                    progress_bar.set_postfix({
                        "loss": f"{loss_meter.avg:.4f}",
                        "lr": f"{lr:.2e}",
                        "grad_norm": f"{grad_norm:.2f}",
                    })

                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint(f"step_{self.global_step}")

                # Evaluation
                if self.eval_dataloader is not None and self.global_step % self.config.eval_steps == 0:
                    eval_loss = self._evaluate()
                    self.logger.info(f"Step {self.global_step} - Eval Loss: {eval_loss:.4f}")
                    self.model.train()

                # Max steps check
                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    break

            # Run callbacks
            for callback in self.callbacks:
                callback(self, step, batch, outputs)

        return loss_meter.avg

    @torch.no_grad()
    def _evaluate(self) -> float:
        """Evaluate model on eval dataset."""
        self.model.eval()
        loss_meter = AverageMeter()

        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(**batch)

            loss_meter.update(outputs.loss.item(), batch["input_ids"].shape[0])

        return loss_meter.avg

    def _save_checkpoint(self, name: str):
        """Save training checkpoint."""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        save_path = checkpoint_dir / f"{name}.pt"
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.epoch,
            step=self.global_step,
            loss=self.best_loss,
            save_path=str(save_path),
            config=vars(self.config),
        )
        self.logger.info(f"Saved checkpoint: {save_path}")

        # Manage checkpoint limit
        self._cleanup_checkpoints(checkpoint_dir)

    def _load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        metadata = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scheduler,
        )
        self.epoch = metadata["epoch"]
        self.global_step = metadata["step"]
        self.best_loss = metadata["loss"]
        self.logger.info(f"Resumed from epoch {self.epoch}, step {self.global_step}")

    def _cleanup_checkpoints(self, checkpoint_dir: Path):
        """Remove old checkpoints beyond save_total_limit."""
        if self.config.save_total_limit <= 0:
            return

        checkpoints = sorted(
            checkpoint_dir.glob("step_*.pt"),
            key=lambda x: x.stat().st_mtime,
        )

        # Keep best and recent checkpoints
        while len(checkpoints) > self.config.save_total_limit:
            old_checkpoint = checkpoints.pop(0)
            old_checkpoint.unlink()
            self.logger.info(f"Removed old checkpoint: {old_checkpoint}")

    def save_model(self, save_path: str):
        """Save model weights only (for inference)."""
        torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Saved model to: {save_path}")


def train_stem_vlm(
    model: nn.Module,
    train_dataset,
    eval_dataset=None,
    config: Optional[TrainingConfig] = None,
    collator=None,
) -> STEMTrainer:
    """Convenience function to train STEM VLM.

    Args:
        model: STEM VLM model
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        config: Training configuration
        collator: Data collator

    Returns:
        Trained STEMTrainer instance
    """
    if config is None:
        config = TrainingConfig()

    from torch.utils.data import DataLoader
    from ..data.collator import MultimodalCollator

    if collator is None:
        collator = MultimodalCollator()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collator,
    )

    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            collate_fn=collator,
        )

    trainer = STEMTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )

    trainer.train()

    return trainer
