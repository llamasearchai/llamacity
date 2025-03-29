import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention between different modalities.
    This allows one modality to attend to another.
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        """
        Initialize cross-attention module.
        
        Args:
            query_dim: Dimensionality of query embedding vectors
            key_dim: Dimensionality of key and value embedding vectors
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in projections
            kdim: Custom key embedding dimension (defaults to key_dim)
            vdim: Custom value embedding dimension (defaults to key_dim)
            head_dim: Dimensionality of each attention head (defaults to query_dim // num_heads)
        """
        super().__init__()
        
        # Set dimensions
        self.query_dim = query_dim
        self.kdim = kdim if kdim is not None else key_dim
        self.vdim = vdim if vdim is not None else key_dim
        self.num_heads = num_heads
        
        # Calculate head dimension
        self.head_dim = head_dim if head_dim is not None else query_dim // num_heads
        assert self.head_dim * num_heads == query_dim, "query_dim must be divisible by num_heads"
        
        # Projection matrices
        self.q_proj = nn.Linear(query_dim, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, num_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, num_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(num_heads * self.head_dim, query_dim, bias=bias)
        
        # Scaling factor for dot product attention
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize/reset model parameters."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Apply cross-attention mechanism.
        
        Args:
            query: Query embeddings [batch_size, seq_len_q, query_dim]
            key: Key embeddings [batch_size, seq_len_k, key_dim]
            value: Value embeddings [batch_size, seq_len_k, key_dim]
            key_mask: Mask for keys [batch_size, seq_len_k]
            attn_mask: Explicit attention mask [batch_size, num_heads, seq_len_q, seq_len_k]
            past_key_value: Cached key-value state from previous decoding steps
            return_attention: Whether to return attention weights
            
        Returns:
            Attention output [batch_size, seq_len_q, query_dim]
            (Optional) Attention weights [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # Project inputs to queries, keys, and values
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape to [batch_size, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        
        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Combine cached key-value states if provided
        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)
            seq_len_k = k.shape[2]
        
        # Compute scaled dot-product attention: (q)(k^T)/sqrt(d) -> softmax -> (attn)(v)
        # [batch_size, num_heads, seq_len_q, head_dim] x [batch_size, num_heads, head_dim, seq_len_k]
        # = [batch_size, num_heads, seq_len_q, seq_len_k]
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scale
        
        # Apply key mask if provided (e.g., padding mask)
        if key_mask is not None:
            # Convert mask to attention mask shape: [batch_size, 1, 1, seq_len_k]
            key_mask = key_mask.view(batch_size, 1, 1, seq_len_k)
            attn_weights = attn_weights.masked_fill(
                key_mask == 0, 
                -1e9
            )
        
        # Apply explicit attention mask if provided
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        
        # Apply softmax to get attention probabilities
        attn_probs = F.softmax(attn_weights, dim=-1)
        
        # Apply dropout
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        # [batch_size, num_heads, seq_len_q, seq_len_k] x [batch_size, num_heads, seq_len_k, head_dim]
        # = [batch_size, num_heads, seq_len_q, head_dim]
        context = torch.matmul(attn_probs, v)
        
        # Transpose back and reshape: [batch_size, seq_len_q, num_heads x head_dim]
        context = context.transpose(1, 2).reshape(batch_size, seq_len_q, -1)
        
        # Final projection
        output = self.out_proj(context)
        
        if return_attention:
            return output, attn_probs
        return output 