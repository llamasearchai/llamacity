import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from src.models.fusion import CrossModalFusion, AttentionalPooling


@dataclass
class FusionConfig:
    """Simple config class for testing fusion modules."""
    fusion_type: str = "cross_attention"  # Options: cross_attention, concat, weighted, gated
    hidden_size: int = 768
    num_attention_heads: int = 8
    dropout: float = 0.1
    learn_weights: bool = True
    use_gate_activation: str = "sigmoid"


class TestCrossModalFusion:
    def test_init_cross_attention(self):
        config = FusionConfig(fusion_type="cross_attention")
        fusion = CrossModalFusion(config)
        
        assert isinstance(fusion, nn.Module)
        assert fusion.fusion_type == "cross_attention"
    
    def test_init_concat(self):
        config = FusionConfig(fusion_type="concat")
        fusion = CrossModalFusion(config)
        
        assert isinstance(fusion, nn.Module)
        assert fusion.fusion_type == "concat"
        assert hasattr(fusion, "projection")
    
    def test_init_weighted(self):
        config = FusionConfig(fusion_type="weighted")
        fusion = CrossModalFusion(config)
        
        assert isinstance(fusion, nn.Module)
        assert fusion.fusion_type == "weighted"
        assert hasattr(fusion, "weights")
    
    def test_init_gated(self):
        config = FusionConfig(fusion_type="gated")
        fusion = CrossModalFusion(config)
        
        assert isinstance(fusion, nn.Module)
        assert fusion.fusion_type == "gated"
        assert hasattr(fusion, "gates")
    
    def test_forward_cross_attention(self):
        config = FusionConfig(fusion_type="cross_attention")
        fusion = CrossModalFusion(config)
        
        batch_size = 2
        text_seq_len = 10
        image_seq_len = 16
        audio_seq_len = 20
        hidden_size = config.hidden_size
        
        text_features = torch.randn(batch_size, text_seq_len, hidden_size)
        image_features = torch.randn(batch_size, image_seq_len, hidden_size)
        audio_features = torch.randn(batch_size, audio_seq_len, hidden_size)
        
        text_attention_mask = torch.ones(batch_size, text_seq_len)
        image_attention_mask = torch.ones(batch_size, image_seq_len)
        audio_attention_mask = torch.ones(batch_size, audio_seq_len)
        
        output = fusion(
            text_features, 
            image_features, 
            audio_features,
            text_attention_mask,
            image_attention_mask,
            audio_attention_mask
        )
        
        # Check that output shape is correct (should be batch_size, seq_len, hidden_size)
        # where seq_len is the combined length of all modalities or a fixed size based on fusion type
        assert output.shape[0] == batch_size
        assert output.shape[2] == hidden_size
    
    def test_forward_concat(self):
        config = FusionConfig(fusion_type="concat")
        fusion = CrossModalFusion(config)
        
        batch_size = 2
        text_seq_len = 10
        image_seq_len = 16
        audio_seq_len = 20
        hidden_size = config.hidden_size
        
        text_features = torch.randn(batch_size, text_seq_len, hidden_size)
        image_features = torch.randn(batch_size, image_seq_len, hidden_size)
        audio_features = torch.randn(batch_size, audio_seq_len, hidden_size)
        
        output = fusion(text_features, image_features, audio_features)
        
        assert output.shape[0] == batch_size
        assert output.shape[2] == hidden_size
    
    def test_forward_weighted(self):
        config = FusionConfig(fusion_type="weighted")
        fusion = CrossModalFusion(config)
        
        batch_size = 2
        text_seq_len = 10
        image_seq_len = 10  # Same length for simplicity
        audio_seq_len = 10  # Same length for simplicity
        hidden_size = config.hidden_size
        
        text_features = torch.randn(batch_size, text_seq_len, hidden_size)
        image_features = torch.randn(batch_size, image_seq_len, hidden_size)
        audio_features = torch.randn(batch_size, audio_seq_len, hidden_size)
        
        output = fusion(text_features, image_features, audio_features)
        
        assert output.shape == (batch_size, text_seq_len, hidden_size)
    
    def test_forward_gated(self):
        config = FusionConfig(fusion_type="gated")
        fusion = CrossModalFusion(config)
        
        batch_size = 2
        seq_len = 10  # Same length for simplicity
        hidden_size = config.hidden_size
        
        text_features = torch.randn(batch_size, seq_len, hidden_size)
        image_features = torch.randn(batch_size, seq_len, hidden_size)
        audio_features = torch.randn(batch_size, seq_len, hidden_size)
        
        output = fusion(text_features, image_features, audio_features)
        
        assert output.shape == (batch_size, seq_len, hidden_size)


class TestAttentionalPooling:
    def test_init(self):
        hidden_size = 768
        num_heads = 8
        
        pooling = AttentionalPooling(
            hidden_size=hidden_size,
            num_heads=num_heads
        )
        
        assert isinstance(pooling, nn.Module)
        assert pooling.num_heads == num_heads
        assert hasattr(pooling, "query")
    
    def test_forward(self):
        batch_size = 2
        seq_len = 10
        hidden_size = 768
        num_heads = 8
        
        pooling = AttentionalPooling(
            hidden_size=hidden_size,
            num_heads=num_heads
        )
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)
        
        output = pooling(hidden_states, attention_mask)
        
        # Output should be [batch_size, hidden_size]
        assert output.shape == (batch_size, hidden_size)
    
    def test_attention_mask(self):
        batch_size = 2
        seq_len = 10
        hidden_size = 768
        num_heads = 8
        
        pooling = AttentionalPooling(
            hidden_size=hidden_size,
            num_heads=num_heads
        )
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Create a mask where the last 5 tokens are masked
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, -5:] = 0
        
        # Get outputs with and without mask
        output_no_mask = pooling(hidden_states)
        output_with_mask = pooling(hidden_states, attention_mask)
        
        # Outputs should be different
        assert not torch.allclose(output_no_mask, output_with_mask, atol=1e-5) 