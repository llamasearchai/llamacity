#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for multi-modal transformer models.
Supports different tasks such as image-text retrieval, visual question answering, and image captioning.
"""

import argparse
import json
import logging
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add the parent directory to Python path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import ModelConfig
from src.data.dataset import MultiModalDataset
from src.losses import ContrastiveLoss, CrossEntropyLoss
from src.model_factory import create_model
from src.utils.checkpointing import load_checkpoint, save_checkpoint
from src.utils.logging import log_metrics, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a multi-modal transformer model"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to model configuration file"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["retrieval", "vqa", "captioning"],
        default="retrieval",
        help="Task to train the model on",
    )
    parser.add_argument(
        "--train_data", type=str, required=True, help="Path to training data"
    )
    parser.add_argument(
        "--val_data", type=str, required=True, help="Path to validation data"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Directory to save outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=64, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--warmup_steps", type=int, default=1000, help="Number of warmup steps"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for gradient clipping",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--log_interval", type=int, default=100, help="Logging interval (in steps)"
    )
    parser.add_argument(
        "--eval_interval", type=int, default=1000, help="Evaluation interval (in steps)"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1000,
        help="Checkpoint saving interval (in steps)",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Use mixed precision training"
    )

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_optimizer_and_scheduler(
    model: nn.Module, args: argparse.Namespace, num_training_steps: int
) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    """
    Create optimizer and learning rate scheduler.

    Args:
        model: Model to optimize
        args: Training arguments
        num_training_steps: Total number of training steps

    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Create parameter groups for weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    # Create optimizer
    optimizer = optim.AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8
    )

    # Create learning rate scheduler
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_steps
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - args.warmup_steps,
        eta_min=0.1 * args.learning_rate,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_steps],
    )

    return optimizer, scheduler


def get_loss_function(task: str, config: ModelConfig) -> nn.Module:
    """
    Get the appropriate loss function for the task.

    Args:
        task: Task name
        config: Model configuration

    Returns:
        Loss function module
    """
    if task == "retrieval":
        task_params = config.task_specific_params.get("image_text_retrieval", {})
        temperature = task_params.get("temperature", 0.07)
        margin = task_params.get("margin", 0.2)
        return ContrastiveLoss(temperature=temperature, margin=margin)

    elif task == "vqa":
        task_params = config.task_specific_params.get("visual_question_answering", {})
        answer_vocab_size = task_params.get("answer_vocab_size", 3000)
        return CrossEntropyLoss(num_classes=answer_vocab_size)

    elif task == "captioning":
        return CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the padding token

    else:
        raise ValueError(f"Unsupported task: {task}")


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    args: argparse.Namespace,
    step: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    """
    Perform a single training step.

    Args:
        model: Model to train
        batch: Batch of data
        loss_fn: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        args: Training arguments
        step: Current step
        scaler: Gradient scaler for mixed precision training

    Returns:
        Dictionary of metrics
    """
    model.train()

    # Move batch to device
    batch = {k: v.to(model.device) for k, v in batch.items()}

    # Forward pass with mixed precision if enabled
    if args.fp16 and scaler is not None:
        with torch.cuda.amp.autocast():
            outputs = model(**batch)
            loss = loss_fn(outputs, batch)
    else:
        outputs = model(**batch)
        loss = loss_fn(outputs, batch)

    # Scale loss for gradient accumulation
    loss = loss / args.gradient_accumulation_steps

    # Backward pass with mixed precision if enabled
    if args.fp16 and scaler is not None:
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (step + 1) % args.gradient_accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
    else:
        loss.backward()

        # Gradient accumulation
        if (step + 1) % args.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    # Compute metrics
    metrics = {"loss": loss.item() * args.gradient_accumulation_steps}

    # Add task-specific metrics
    if args.task == "retrieval" and "scores" in outputs:
        metrics.update(compute_retrieval_metrics(outputs["scores"]))
    elif args.task == "vqa" and "logits" in outputs:
        metrics.update(compute_vqa_metrics(outputs["logits"], batch["answers"]))
    elif args.task == "captioning" and "logits" in outputs:
        metrics.update(
            compute_captioning_metrics(outputs["logits"], batch["caption_ids"])
        )

    return metrics


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """
    Evaluate the model on the validation set.

    Args:
        model: Model to evaluate
        dataloader: Validation dataloader
        loss_fn: Loss function
        args: Training arguments

    Returns:
        Dictionary of metrics
    """
    model.eval()

    total_loss = 0.0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(model.device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = loss_fn(outputs, batch)

            # Accumulate loss
            total_loss += loss.item()

            # Store outputs and targets for computing metrics
            if args.task == "retrieval" and "scores" in outputs:
                all_outputs.append(outputs["scores"].cpu())
            elif args.task == "vqa" and "logits" in outputs:
                all_outputs.append(outputs["logits"].cpu())
                all_targets.append(batch["answers"].cpu())
            elif args.task == "captioning" and "logits" in outputs:
                all_outputs.append(outputs["logits"].cpu())
                all_targets.append(batch["caption_ids"].cpu())

    # Compute average loss
    avg_loss = total_loss / len(dataloader)
    metrics = {"loss": avg_loss}

    # Compute task-specific metrics
    if args.task == "retrieval" and all_outputs:
        all_scores = torch.cat(all_outputs, dim=0)
        metrics.update(compute_retrieval_metrics(all_scores))
    elif args.task in ["vqa", "captioning"] and all_outputs and all_targets:
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        if args.task == "vqa":
            metrics.update(compute_vqa_metrics(all_outputs, all_targets))
        else:  # captioning
            metrics.update(compute_captioning_metrics(all_outputs, all_targets))

    return metrics


def compute_retrieval_metrics(scores: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics for image-text retrieval.

    Args:
        scores: Similarity scores [batch_size, batch_size]

    Returns:
        Dictionary of metrics
    """
    batch_size = scores.size(0)

    # Image-to-text retrieval
    i2t_ranks = torch.zeros(batch_size)
    for i in range(batch_size):
        # For each image, compute the rank of the matching text
        i2t_ranks[i] = (scores[i] >= scores[i, i]).sum().item() - 1

    # Text-to-image retrieval
    t2i_ranks = torch.zeros(batch_size)
    for i in range(batch_size):
        # For each text, compute the rank of the matching image
        t2i_ranks[i] = (scores[:, i] >= scores[i, i]).sum().item() - 1

    # Compute recall metrics
    metrics = {}
    for k in [1, 5, 10]:
        metrics[f"i2t_r@{k}"] = (i2t_ranks < k).float().mean().item() * 100
        metrics[f"t2i_r@{k}"] = (t2i_ranks < k).float().mean().item() * 100
        metrics[f"r@{k}"] = (metrics[f"i2t_r@{k}"] + metrics[f"t2i_r@{k}"]) / 2

    return metrics


def compute_vqa_metrics(
    logits: torch.Tensor, targets: torch.Tensor
) -> Dict[str, float]:
    """
    Compute metrics for visual question answering.

    Args:
        logits: Predicted logits [batch_size, num_classes]
        targets: Target answer indices [batch_size]

    Returns:
        Dictionary of metrics
    """
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions == targets).float().mean().item() * 100

    return {"accuracy": accuracy}


def compute_captioning_metrics(
    logits: torch.Tensor, targets: torch.Tensor
) -> Dict[str, float]:
    """
    Compute metrics for image captioning.

    Args:
        logits: Predicted logits [batch_size, seq_len, vocab_size]
        targets: Target caption indices [batch_size, seq_len]

    Returns:
        Dictionary of metrics
    """
    # Reshape logits for cross-entropy calculation
    batch_size, seq_len, vocab_size = logits.size()
    logits = logits.reshape(-1, vocab_size)
    targets = targets.reshape(-1)

    # Compute accuracy (ignoring padding tokens)
    mask = targets != 0  # Assuming 0 is the padding token
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions[mask] == targets[mask]).float().mean().item() * 100

    return {"accuracy": accuracy}


def main():
    args = parse_args()

    # Set up logging
    setup_logging(args.output_dir)
    logger = logging.getLogger(__name__)
    logger.info(f"Arguments: {args}")

    # Set random seed
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save arguments
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load model configuration
    logger.info(f"Loading model configuration from {args.config}")
    config = ModelConfig.from_json_file(args.config)

    # Create model
    logger.info("Creating model")
    model = create_model(args.config)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.device = device

    # Log model information
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {num_params:,} parameters")

    # Create datasets and dataloaders
    logger.info("Creating datasets and dataloaders")
    train_dataset = MultiModalDataset(
        data_path=args.train_data, task=args.task, split="train"
    )

    val_dataset = MultiModalDataset(
        data_path=args.val_data, task=args.task, split="val"
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Calculate training steps
    num_training_steps = len(train_dataloader) * args.num_epochs

    # Create optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model, args=args, num_training_steps=num_training_steps
    )

    # Create loss function
    loss_fn = get_loss_function(args.task, config)

    # Initialize mixed precision training if enabled
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    best_val_metric = float("inf") if args.task == "captioning" else 0.0

    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint = load_checkpoint(args.resume_from)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        best_val_metric = checkpoint["best_val_metric"]

        if args.fp16 and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # Training loop
    logger.info("Starting training")

    for epoch in range(start_epoch, args.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")

        # Training
        model.train()
        train_metrics = {}

        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            # Perform training step
            metrics = train_step(
                model=model,
                batch=batch,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                step=global_step,
                scaler=scaler,
            )

            # Update metrics
            for k, v in metrics.items():
                if k not in train_metrics:
                    train_metrics[k] = []
                train_metrics[k].append(v)

            # Log metrics
            if (global_step + 1) % args.log_interval == 0:
                log_metrics(
                    metrics={
                        k: np.mean(v[-args.log_interval :])
                        for k, v in train_metrics.items()
                    },
                    step=global_step,
                    prefix="train",
                )

            # Evaluate
            if (global_step + 1) % args.eval_interval == 0:
                logger.info(f"Evaluating at step {global_step + 1}")
                val_metrics = evaluate(
                    model=model, dataloader=val_dataloader, loss_fn=loss_fn, args=args
                )

                log_metrics(metrics=val_metrics, step=global_step, prefix="val")

                # Save best model
                if args.task == "captioning":
                    # For captioning, lower loss is better
                    current_metric = val_metrics["loss"]
                    is_best = current_metric < best_val_metric
                else:
                    # For retrieval and VQA, higher metric is better
                    current_metric = val_metrics.get(
                        "r@1", val_metrics.get("accuracy", 0.0)
                    )
                    is_best = current_metric > best_val_metric

                if is_best:
                    best_val_metric = current_metric
                    logger.info(f"New best model with {best_val_metric:.4f}")

                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        global_step=global_step,
                        best_val_metric=best_val_metric,
                        scaler=scaler,
                        output_dir=args.output_dir,
                        is_best=True,
                    )

            # Save checkpoint
            if (global_step + 1) % args.save_interval == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    best_val_metric=best_val_metric,
                    scaler=scaler,
                    output_dir=args.output_dir,
                    is_best=False,
                )

            global_step += 1

        # End of epoch
        logger.info(f"End of epoch {epoch + 1}")

        # Log epoch metrics
        log_metrics(
            metrics={k: np.mean(v) for k, v in train_metrics.items()},
            step=epoch,
            prefix="epoch",
        )

        # Save checkpoint at the end of each epoch
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            global_step=global_step,
            best_val_metric=best_val_metric,
            scaler=scaler,
            output_dir=args.output_dir,
            is_best=False,
        )

    # Final evaluation
    logger.info("Final evaluation")
    val_metrics = evaluate(
        model=model, dataloader=val_dataloader, loss_fn=loss_fn, args=args
    )

    log_metrics(metrics=val_metrics, step=global_step, prefix="final")

    logger.info("Training completed")


if __name__ == "__main__":
    main()
