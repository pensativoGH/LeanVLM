"""Trainer for TinyVLM with step-based training."""

import os
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Any, Optional
from tqdm import tqdm
from pathlib import Path

from src.model import TinyVLM
from src.data import VLMProcessor, create_dataloader


class Trainer:
    """
    Trainer for TinyVLM with step-based training.

    Features:
    - Mixed precision training (bf16)
    - Gradient accumulation
    - Cosine learning rate scheduling with warmup
    - TensorBoard logging
    - Checkpointing
    """

    def __init__(
        self,
        model: TinyVLM,
        processor: VLMProcessor,
        config: Dict[str, Any],
        stage: int = 1,
    ):
        """
        Initialize the trainer.

        Args:
            model: TinyVLM model
            processor: VLMProcessor for data
            config: Training configuration
            stage: Training stage (1 or 2)
        """
        self.model = model
        self.processor = processor
        self.config = config
        self.stage = stage

        # Get stage-specific config
        stage_key = f"stage{stage}"
        self.stage_config = config[stage_key]

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Enable TF32 for faster training on Ampere GPUs
        if config["hardware"]["tf32"]:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')

        # Setup precision
        self.precision = config["hardware"]["precision"]
        self.use_amp = self.precision == "bf16" and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp and self.precision != "bf16" else None

        # Configure model freezing based on stage
        self._configure_freezing()

        # Create optimizer (only for trainable parameters)
        self.optimizer = self._create_optimizer()

        # Create learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Create dataloader with stage-specific config
        use_chat_template = self.stage_config.get("use_chat_template", True)
        dataset_type = self.stage_config.get("dataset", "cauldron")
        subsets = self.stage_config.get("subsets", None)
        max_samples = self.stage_config.get("max_samples", None)

        self.dataloader = create_dataloader(
            processor=processor,
            batch_size=self.stage_config["batch_size"],
            num_workers=config["data"]["num_workers"],
            use_chat_template=use_chat_template,
            dataset_type=dataset_type,
            subsets=subsets,
            max_samples=max_samples,
        )

        # Logging
        log_dir = Path(config["paths"]["log_dir"]) / f"stage{stage}"
        self.writer = SummaryWriter(log_dir=str(log_dir))

        # CSV log file (easy to download and view)
        self.csv_log_path = log_dir / "training_log.csv"
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.csv_log_path, "w") as f:
            f.write("step,loss,perplexity,learning_rate\n")

        # Checkpoint directory
        self.checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.global_step = 0
        self.accumulated_loss = 0.0

    def _configure_freezing(self):
        """Configure which components are frozen based on stage."""
        # Vision encoder: frozen by default, can be unfrozen in Stage 2
        freeze_vision = self.stage_config.get("freeze_vision", True)
        if freeze_vision:
            self.model.freeze_vision_encoder()
            print("Vision encoder: frozen")
        else:
            self.model.unfreeze_vision_encoder()
            print("Vision encoder: trainable (like nanoVLM)")

        freeze_lm = self.stage_config["freeze_lm"]

        if freeze_lm == "partial":
            # Partial LM freeze: train embeddings + first block
            unfreeze_layers = self.stage_config.get("unfreeze_lm_layers", ["embed", 0])
            self.model.freeze_language_model_partial(unfreeze_layers)
            print(f"Stage 1: Training projector + LM layers {unfreeze_layers}")
        elif freeze_lm:
            self.model.freeze_language_model()
            print("Stage 1: Training projector only")
        else:
            self.model.unfreeze_language_model()
            print("Stage 2: Training projector + language model")

        total, trainable = self.model.count_parameters()
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")

    def _create_optimizer(self) -> AdamW:
        """Create optimizer with per-component learning rates.

        Uses different learning rates for each component:
        - Projector: high LR (newly initialized)
        - Vision encoder: low LR (pre-trained, only when unfrozen)
        - Language model: low LR (pre-trained)
        Following nanoVLM's approach for better convergence.
        """
        # Get per-component learning rates from config
        lr_config = self.config.get("learning_rates", {})
        projector_lr = lr_config.get("projector", 5e-3)
        vision_lr = lr_config.get("vision_encoder", 5e-5)
        lm_lr = lr_config.get("language_model", 5e-5)

        # Group parameters by component
        projector_params = []
        vision_params = []
        lm_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "projector" in name:
                projector_params.append(param)
            elif "vision_encoder" in name:
                vision_params.append(param)
            elif "language_model" in name:
                lm_params.append(param)

        # Create parameter groups with different learning rates
        param_groups = []
        if projector_params:
            param_groups.append({
                "params": projector_params,
                "lr": projector_lr,
                "name": "projector",
            })
        if vision_params:
            param_groups.append({
                "params": vision_params,
                "lr": vision_lr,
                "name": "vision_encoder",
            })
        if lm_params:
            param_groups.append({
                "params": lm_params,
                "lr": lm_lr,
                "name": "language_model",
            })

        print(f"Optimizer parameter groups:")
        print(f"  Projector: {len(projector_params)} tensors, lr={projector_lr}")
        print(f"  Vision Encoder: {len(vision_params)} tensors, lr={vision_lr}")
        print(f"  Language Model: {len(lm_params)} tensors, lr={lm_lr}")

        return AdamW(
            param_groups,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup.

        Works with per-component learning rates - each param group
        gets its own warmup and cosine decay proportional to its base LR.
        """
        warmup_steps = self.stage_config["warmup_steps"]
        max_steps = self.stage_config["max_steps"]

        # Linear warmup (applies to all param groups proportionally)
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        # Cosine decay after warmup
        # eta_min is set proportionally for each param group
        # Using the minimum of all group LRs * 0.1 as baseline
        min_lr = min(pg["lr"] for pg in self.optimizer.param_groups) * 0.1
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_steps - warmup_steps,
            eta_min=min_lr,
        )

        # Combine warmup and cosine
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        return scheduler

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Execute a single training step.

        Args:
            batch: Batch of data

        Returns:
            Loss value
        """
        # Move batch to device
        pixel_values = batch["pixel_values"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Convert to bf16 if using mixed precision
        if self.use_amp:
            pixel_values = pixel_values.to(torch.bfloat16)

        # Forward pass
        with autocast(dtype=torch.bfloat16, enabled=self.use_amp):
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            # Scale loss for gradient accumulation
            loss = loss / self.stage_config["gradient_accumulation_steps"]

        # Backward pass
        loss.backward()

        return loss.item() * self.stage_config["gradient_accumulation_steps"]

    def train(self, resume_from: Optional[str] = None):
        """
        Main training loop.

        Args:
            resume_from: Path to checkpoint to resume from
        """
        if resume_from:
            self.load_checkpoint(resume_from)

        max_steps = self.stage_config["max_steps"]
        grad_accum_steps = self.stage_config["gradient_accumulation_steps"]
        log_steps = self.stage_config["log_steps"]
        save_steps = self.stage_config["save_steps"]

        self.model.train()

        print("Creating data iterator...")
        data_iter = iter(self.dataloader)

        print("Fetching first batch (this may take a minute)...")
        first_batch = next(data_iter)
        print(f"First batch loaded! Shape: {first_batch['pixel_values'].shape}")

        # Put first batch back by creating a chain
        from itertools import chain
        data_iter = chain([first_batch], data_iter)

        progress_bar = tqdm(
            total=max_steps,
            initial=self.global_step,
            desc=f"Stage {self.stage} Training",
        )

        print("Starting training loop...")

        while self.global_step < max_steps:
            self.accumulated_loss = 0.0

            # Gradient accumulation loop
            for accum_step in range(grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    # Restart the iterator if we run out of data
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)

                loss = self.train_step(batch)
                self.accumulated_loss += loss / grad_accum_steps

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.get_trainable_parameters(),
                max_norm=1.0,
            )

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            self.global_step += 1
            progress_bar.update(1)

            # Logging
            if self.global_step % log_steps == 0:
                self._log_metrics()

            # Save checkpoint
            if self.global_step % save_steps == 0:
                self.save_checkpoint(f"stage{self.stage}_step{self.global_step}.pt")

            progress_bar.set_postfix({
                "loss": f"{self.accumulated_loss:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

        progress_bar.close()

        # Save final checkpoint
        self.save_checkpoint(f"stage{self.stage}_final.pt")
        print(f"Training complete! Final checkpoint saved.")

    def _log_metrics(self):
        """Log metrics to TensorBoard and CSV."""
        loss = self.accumulated_loss
        lr = self.scheduler.get_last_lr()[0]
        ppl = math.exp(min(loss, 20))  # Cap to avoid overflow

        # TensorBoard
        self.writer.add_scalar("train/loss", loss, self.global_step)
        self.writer.add_scalar("train/learning_rate", lr, self.global_step)
        self.writer.add_scalar("train/perplexity", ppl, self.global_step)

        # CSV (append)
        with open(self.csv_log_path, "a") as f:
            f.write(f"{self.global_step},{loss:.6f},{ppl:.4f},{lr:.2e}\n")

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": {
                "vision_encoder": self.model.vision_encoder.state_dict(),
                "projector": self.model.projector.state_dict(),
                "language_model": self.model.language_model.model.state_dict(),
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "stage": self.stage,
            "config": self.config,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model weights
        self.model.vision_encoder.load_state_dict(
            checkpoint["model_state_dict"]["vision_encoder"]
        )
        self.model.projector.load_state_dict(
            checkpoint["model_state_dict"]["projector"]
        )
        self.model.language_model.model.load_state_dict(
            checkpoint["model_state_dict"]["language_model"]
        )

        # Only load optimizer/scheduler if same stage
        if checkpoint.get("stage") == self.stage:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.global_step = checkpoint["global_step"]
            print(f"Resumed from step {self.global_step}")
        else:
            print(f"Loaded model weights from stage {checkpoint.get('stage')}")

    def close(self):
        """Clean up resources."""
        self.writer.close()
