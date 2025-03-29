import os
import sys
import pytest
import torch
import numpy as np
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import ModelConfig, MultiModalConfig, EncoderConfig
from src.models.layers import PositionalEncoding, FeedForward, LayerNorm
from src.models.attention import MultiHeadCrossAttention, LinearAttention, SparseMultiHeadAttention
from src.models.fusion import CrossModalFusion, AttentionalPooling


@pytest.fixture(scope="session")
def data_dir():
    """Return the path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def dummy_text_features():
    """Return dummy text features for testing."""
    batch_size = 2
    seq_len = 10
    hidden_size = 768
    return torch.randn(batch_size, seq_len, hidden_size)


@pytest.fixture
def dummy_image_features():
    """Return dummy image features for testing."""
    batch_size = 2
    num_patches = 16
    hidden_size = 768
    return torch.randn(batch_size, num_patches, hidden_size)


@pytest.fixture
def dummy_audio_features():
    """Return dummy audio features for testing."""
    batch_size = 2
    seq_len = 20
    hidden_size = 768
    return torch.randn(batch_size, seq_len, hidden_size)


@pytest.fixture
def dummy_attention_mask():
    """Return dummy attention mask for testing."""
    batch_size = 2
    seq_len = 10
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    # Add some padding in the sequence
    mask[:, -2:] = 0
    return mask


@pytest.fixture
def model_config():
    """Return a model configuration for testing."""
    return ModelConfig.from_dict({
        "model_type": "encoder_only",
        "hidden_size": 768,
        "vocab_size": 30522,
        "multi_modal_config": {
            "text_encoder": {
                "hidden_size": 768,
                "num_hidden_layers": 2,
                "num_attention_heads": 8
            },
            "image_encoder": {
                "hidden_size": 768,
                "num_hidden_layers": 2,
                "num_attention_heads": 8
            },
            "fusion_config": {
                "fusion_type": "cross_attention",
                "hidden_size": 768,
                "num_attention_heads": 8
            }
        }
    }) 