import logging
import os
import shutil
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    global_step: int = 0,
    best_val_metric: float = 0.0,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    output_dir: str = "checkpoints",
    is_best: bool = False,
    filename: str = "checkpoint.pt",
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Learning rate scheduler to save
        epoch: Current epoch
        global_step: Current global step
        best_val_metric: Best validation metric so far
        scaler: Gradient scaler for mixed precision training
        output_dir: Directory to save checkpoint
        is_best: Whether this is the best checkpoint so far
        filename: Name of the checkpoint file
    """
    logger = logging.getLogger(__name__)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create checkpoint dictionary
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_metric": best_val_metric,
    }

    # Add scheduler state if provided
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    # Add scaler state if provided
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    # Save checkpoint
    checkpoint_path = os.path.join(output_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Save as best checkpoint if specified
    if is_best:
        best_path = os.path.join(output_dir, "best_checkpoint.pt")
        shutil.copyfile(checkpoint_path, best_path)
        logger.info(f"Saved best checkpoint to {best_path}")

    # Save model separately for easier loading
    model_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model to {model_path}")


def load_checkpoint(
    checkpoint_path: str, map_location: Optional[Union[str, torch.device]] = None
) -> Dict[str, Any]:
    """
    Load checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        map_location: Device to map tensors to

    Returns:
        Checkpoint dictionary
    """
    logger = logging.getLogger(__name__)

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    return checkpoint


def load_model_from_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = True,
    map_location: Optional[Union[str, torch.device]] = None,
) -> nn.Module:
    """
    Load model from checkpoint.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        strict: Whether to strictly enforce that the keys in state_dict match the keys in model
        map_location: Device to map tensors to

    Returns:
        Model with loaded weights
    """
    logger = logging.getLogger(__name__)

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Load checkpoint
    logger.info(f"Loading model from {checkpoint_path}")

    # Check if it's a full checkpoint or just model weights
    if checkpoint_path.endswith(".bin"):
        # Load model weights directly
        state_dict = torch.load(checkpoint_path, map_location=map_location)
    else:
        # Load from full checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        state_dict = checkpoint["model_state_dict"]

    # Load state dict into model
    model.load_state_dict(state_dict, strict=strict)

    return model


def save_model_for_inference(
    model: nn.Module, output_dir: str, config_path: Optional[str] = None
) -> None:
    """
    Save model for inference.

    Args:
        model: Model to save
        output_dir: Directory to save model
        config_path: Path to model configuration file
    """
    logger = logging.getLogger(__name__)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save model weights
    model_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model for inference to {model_path}")

    # Copy configuration file if provided
    if config_path is not None and os.path.exists(config_path):
        config_output_path = os.path.join(output_dir, "config.json")
        shutil.copyfile(config_path, config_output_path)
        logger.info(f"Copied configuration to {config_output_path}")


def cleanup_checkpoints(
    output_dir: str, keep_last_n: int = 5, keep_best: bool = True
) -> None:
    """
    Clean up old checkpoints to save disk space.

    Args:
        output_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        keep_best: Whether to keep the best checkpoint
    """
    logger = logging.getLogger(__name__)

    # Get all checkpoint files
    checkpoint_files = [
        f
        for f in os.listdir(output_dir)
        if f.startswith("checkpoint") and f.endswith(".pt")
    ]

    # Sort by modification time (newest first)
    checkpoint_files.sort(
        key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True
    )

    # Keep the best checkpoint if specified
    if keep_best and "best_checkpoint.pt" in checkpoint_files:
        checkpoint_files.remove("best_checkpoint.pt")

    # Remove old checkpoints
    if len(checkpoint_files) > keep_last_n:
        for checkpoint_file in checkpoint_files[keep_last_n:]:
            checkpoint_path = os.path.join(output_dir, checkpoint_file)
            os.remove(checkpoint_path)
            logger.info(f"Removed old checkpoint: {checkpoint_path}")

    logger.info(f"Kept {min(keep_last_n, len(checkpoint_files))} recent checkpoints")
