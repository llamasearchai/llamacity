import pytest
import torch
import torch.nn as nn
from src.models.layers import PositionalEncoding, FeedForward, LayerNorm
from src.models.layers.positional_encoding import RotaryPositionalEncoding, RelativePositionalEncoding


class TestPositionalEncoding:
    def test_init(self):
        d_model = 128
        max_len = 100
        dropout = 0.1
        pe = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)
        
        assert isinstance(pe, nn.Module)
        assert pe.dropout.p == dropout
        assert pe.pe.shape == (1, max_len, d_model)
    
    def test_forward(self):
        batch_size = 2
        seq_len = 10
        d_model = 128
        
        pe = PositionalEncoding(d_model=d_model)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = pe(x)
        
        assert output.shape == x.shape
        assert torch.allclose(output, x + pe.pe[:, :seq_len, :], atol=1e-5)
    
    def test_learnable(self):
        d_model = 128
        max_len = 100
        
        pe = PositionalEncoding(d_model=d_model, max_len=max_len, learnable=True)
        
        # Check if parameters are learnable
        assert sum(p.numel() for p in pe.parameters() if p.requires_grad) > 0
        
        # Check forward
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, d_model)
        output = pe(x)
        
        assert output.shape == x.shape


class TestRotaryPositionalEncoding:
    def test_init(self):
        dim = 128
        max_len = 100
        rpe = RotaryPositionalEncoding(dim=dim, max_len=max_len)
        
        assert isinstance(rpe, nn.Module)
        assert rpe.dim == dim
        assert rpe.max_len == max_len
    
    def test_forward(self):
        batch_size = 2
        seq_len = 10
        dim = 128
        
        rpe = RotaryPositionalEncoding(dim=dim)
        x = torch.randn(batch_size, seq_len, dim)
        
        output = rpe(x)
        
        assert output.shape == x.shape
        # Output should be different from input
        assert not torch.allclose(output, x, atol=1e-5)


class TestFeedForward:
    def test_init(self):
        d_model = 128
        d_ff = 512
        dropout = 0.1
        
        ff = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        
        assert isinstance(ff, nn.Module)
        assert ff.w1.in_features == d_model
        assert ff.w1.out_features == d_ff
        assert ff.w2.in_features == d_ff
        assert ff.w2.out_features == d_model
        assert ff.dropout.p == dropout
    
    def test_forward(self):
        batch_size = 2
        seq_len = 10
        d_model = 128
        d_ff = 512
        
        ff = FeedForward(d_model=d_model, d_ff=d_ff)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = ff(x)
        
        assert output.shape == x.shape
    
    @pytest.mark.parametrize("activation", ["relu", "gelu", "swish", "silu"])
    def test_activations(self, activation):
        d_model = 128
        d_ff = 512
        
        ff = FeedForward(d_model=d_model, d_ff=d_ff, activation=activation)
        x = torch.randn(2, 10, d_model)
        
        output = ff(x)
        
        assert output.shape == x.shape


class TestLayerNorm:
    def test_init(self):
        normalized_shape = 128
        eps = 1e-5
        
        ln = LayerNorm(normalized_shape=normalized_shape, eps=eps)
        
        assert isinstance(ln, nn.Module)
        assert ln.weight.shape == (normalized_shape,)
        assert ln.bias.shape == (normalized_shape,)
        assert ln.eps == eps
    
    def test_forward(self):
        batch_size = 2
        seq_len = 10
        normalized_shape = 128
        
        ln = LayerNorm(normalized_shape=normalized_shape)
        x = torch.randn(batch_size, seq_len, normalized_shape)
        
        output = ln(x)
        
        assert output.shape == x.shape
        
        # Test that normalization happened
        mean = output.mean(dim=-1, keepdim=True)
        var = ((output - mean) ** 2).mean(dim=-1, keepdim=True)
        
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-5) 