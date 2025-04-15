#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script demonstrating how to create and use a multi-modal transformer model
using the model factory utilities.
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the parent directory to Python path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import ModelConfig, MultiModalConfig
from src.model_factory import ModelFactory, create_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create and test a multi-modal transformer model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_text_image_model.json",
        help="Path to model configuration file",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for testing"
    )
    parser.add_argument(
        "--seq_length", type=int, default=20, help="Sequence length for testing"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="models/test_model",
        help="Directory to save the model configuration",
    )

    return parser.parse_args()


def create_dummy_inputs(
    batch_size: int, seq_length: int, image_size: int = 224, hidden_dim: int = 768
) -> Dict[str, torch.Tensor]:
    """
    Create dummy inputs for the multi-modal transformer model.

    Args:
        batch_size: Batch size
        seq_length: Sequence length for text
        image_size: Size of the image (height and width)
        hidden_dim: Hidden dimension for features

    Returns:
        Dictionary of dummy inputs
    """
    # Create dummy text inputs
    text_inputs = torch.randint(0, 1000, (batch_size, seq_length))
    text_attention_mask = torch.ones_like(text_inputs)

    # Add random padding to text
    for i in range(batch_size):
        padding_length = np.random.randint(0, seq_length // 4)
        if padding_length > 0:
            text_attention_mask[i, -padding_length:] = 0

    # Create dummy image inputs (already processed through vision encoder)
    # Shape: [batch_size, seq_len, hidden_dim]
    # where seq_len is the number of patches in the image
    num_patches = (image_size // 16) ** 2  # Assuming 16x16 patches
    image_features = torch.randn(batch_size, num_patches, hidden_dim)
    image_attention_mask = torch.ones(batch_size, num_patches)

    return {
        "text_inputs": text_inputs,
        "text_attention_mask": text_attention_mask,
        "image_features": image_features,
        "image_attention_mask": image_attention_mask,
    }


def main():
    args = parse_args()

    print(f"Loading model configuration from {args.config}")

    # Create model from configuration file
    model = create_model(args.config)

    print(f"Created model:")
    print(f"  Model type: {model.__class__.__name__}")
    print(f"  Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Create dummy inputs
    hidden_size = model.model.config.text_encoder.hidden_size
    inputs = create_dummy_inputs(
        batch_size=args.batch_size, seq_length=args.seq_length, hidden_dim=hidden_size
    )

    # Run forward pass
    print("Running forward pass with dummy inputs...")

    with torch.no_grad():
        outputs = model(
            text_inputs=inputs["text_inputs"],
            text_attention_mask=inputs["text_attention_mask"],
            image_features=inputs["image_features"],
            image_attention_mask=inputs["image_attention_mask"],
        )

    # Print output shapes
    print("\nOutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {tuple(value.shape)}")
        elif isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, torch.Tensor):
                    print(f"    {sub_key}: {tuple(sub_value.shape)}")

    # Save the model configuration
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        config_path = os.path.join(args.save_dir, "config.json")

        # Get the configuration from the model
        config = ModelConfig.from_json_file(args.config)
        config.save_pretrained(args.save_dir)

        print(f"\nSaved model configuration to {config_path}")


if __name__ == "__main__":
    main()
