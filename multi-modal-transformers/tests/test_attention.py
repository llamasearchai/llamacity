import pytest
import torch
import torch.nn as nn
from src.models.attention import MultiHeadCrossAttention, SparseMultiHeadAttention, LinearAttention


class TestMultiHeadCrossAttention:
    def test_init(self):
        query_dim = 128
        key_dim = 256
        num_heads = 8
        dropout = 0.1
        
        attn = MultiHeadCrossAttention(
            query_dim=query_dim,
            key_dim=key_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        assert isinstance(attn, nn.Module)
        assert attn.num_heads == num_heads
        assert attn.q_proj.in_features == query_dim
        assert attn.k_proj.in_features == key_dim
        assert attn.v_proj.in_features == key_dim
        assert attn.dropout.p == dropout
    
    def test_forward(self):
        batch_size = 2
        seq_len_q = 10
        seq_len_k = 15
        query_dim = 128
        key_dim = 256
        num_heads = 8
        
        attn = MultiHeadCrossAttention(
            query_dim=query_dim,
            key_dim=key_dim,
            num_heads=num_heads
        )
        
        query = torch.randn(batch_size, seq_len_q, query_dim)
        key = torch.randn(batch_size, seq_len_k, key_dim)
        value = torch.randn(batch_size, seq_len_k, key_dim)
        
        output = attn(query, key, value)
        
        assert output.shape == (batch_size, seq_len_q, query_dim)
    
    def test_attention_mask(self):
        batch_size = 2
        seq_len_q = 10
        seq_len_k = 15
        query_dim = 128
        key_dim = 256
        num_heads = 8
        
        attn = MultiHeadCrossAttention(
            query_dim=query_dim,
            key_dim=key_dim,
            num_heads=num_heads
        )
        
        query = torch.randn(batch_size, seq_len_q, query_dim)
        key = torch.randn(batch_size, seq_len_k, key_dim)
        value = torch.randn(batch_size, seq_len_k, key_dim)
        
        # Create a key mask where the last 5 tokens are masked
        key_mask = torch.ones(batch_size, seq_len_k)
        key_mask[:, -5:] = 0
        
        # Get outputs with and without mask
        output_no_mask = attn(query, key, value)
        output_with_mask = attn(query, key, value, key_mask=key_mask)
        
        # Outputs should be different
        assert not torch.allclose(output_no_mask, output_with_mask, atol=1e-5)
    
    def test_return_attention(self):
        batch_size = 2
        seq_len_q = 10
        seq_len_k = 15
        query_dim = 128
        key_dim = 256
        num_heads = 8
        
        attn = MultiHeadCrossAttention(
            query_dim=query_dim,
            key_dim=key_dim,
            num_heads=num_heads
        )
        
        query = torch.randn(batch_size, seq_len_q, query_dim)
        key = torch.randn(batch_size, seq_len_k, key_dim)
        value = torch.randn(batch_size, seq_len_k, key_dim)
        
        output, attention = attn(query, key, value, return_attention=True)
        
        assert output.shape == (batch_size, seq_len_q, query_dim)
        assert attention.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        
        # Check that attention weights sum to 1 along the key dimension
        assert torch.allclose(attention.sum(dim=-1), torch.ones_like(attention.sum(dim=-1)), atol=1e-5)


class TestSparseMultiHeadAttention:
    def test_init(self):
        hidden_size = 128
        num_heads = 8
        dropout = 0.1
        window_size = 4
        stride = 2
        
        attn = SparseMultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            window_size=window_size,
            stride=stride
        )
        
        assert isinstance(attn, nn.Module)
        assert attn.num_heads == num_heads
        assert attn.window_size == window_size
        assert attn.stride == stride
        assert attn.head_dim == hidden_size // num_heads
    
    def test_forward(self):
        batch_size = 2
        seq_len = 16
        hidden_size = 128
        num_heads = 8
        window_size = 4
        stride = 2
        
        attn = SparseMultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            window_size=window_size,
            stride=stride
        )
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        output = attn(hidden_states)
        
        assert output.shape == hidden_states.shape


class TestLinearAttention:
    def test_init(self):
        hidden_size = 128
        num_heads = 8
        dropout = 0.1
        dim = 64
        
        attn = LinearAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            dim=dim
        )
        
        assert isinstance(attn, nn.Module)
        assert attn.num_heads == num_heads
        assert attn.dim == dim
        assert attn.dropout.p == dropout
    
    def test_forward(self):
        batch_size = 2
        seq_len = 16
        hidden_size = 128
        num_heads = 8
        dim = 64
        
        attn = LinearAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dim=dim
        )
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        output = attn(hidden_states)
        
        assert output.shape == hidden_states.shape
    
    @pytest.mark.parametrize("kernel_function", ["elu", "relu", "softmax"])
    def test_kernel_functions(self, kernel_function):
        hidden_size = 128
        num_heads = 8
        dim = 64
        
        attn = LinearAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dim=dim,
            kernel_function=kernel_function
        )
        
        hidden_states = torch.randn(2, 16, hidden_size)
        
        output = attn(hidden_states)
        
        assert output.shape == hidden_states.shape 