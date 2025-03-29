import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import einops


class LinearAttention(nn.Module):
    """
    Linear Attention with kernel method for O(N) complexity instead of O(N²).
    Based on the paper "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
    (https://arxiv.org/abs/2006.16236)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        dim: int = 64,
        eps: float = 1e-6,
        causal: bool = False,
        bias: bool = True,
        kernel_function: str = "elu",
    ):
        """
        Initialize linear attention module.
        
        Args:
            hidden_size: Dimensionality of embedding vectors
            num_heads: Number of attention heads
            dropout: Dropout probability
            dim: Feature dimension of the kernel space projection
            eps: Small value for numerical stability
            causal: Whether to use causal masking (for autoregressive models)
            bias: Whether to use bias in projections
            kernel_function: Kernel function to use ('elu', 'relu', 'softmax')
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dim = dim
        self.eps = eps
        self.causal = causal
        self.kernel_function = kernel_function
        
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        # Projection matrices
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        # Feature maps for Q, K if using a kernel method
        if dim > 0:
            self.q_map = nn.Linear(self.head_dim, dim, bias=False)
            self.k_map = nn.Linear(self.head_dim, dim, bias=False)
        
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
        
        if self.dim > 0:
            nn.init.normal_(self.q_map.weight, mean=0, std=self.q_map.weight.shape[0] ** -0.5)
            nn.init.normal_(self.k_map.weight, mean=0, std=self.k_map.weight.shape[0] ** -0.5)
    
    def _apply_kernel_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply kernel function to transform attention scores.
        
        Args:
            x: Input tensor
            
        Returns:
            Transformed tensor
        """
        if self.kernel_function == "elu":
            return F.elu(x) + 1.0
        elif self.kernel_function == "relu":
            return F.relu(x) + self.eps
        elif self.kernel_function == "softmax":
            return F.softmax(x, dim=-1)
        else:
            raise ValueError(f"Unknown kernel function: {self.kernel_function}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Apply linear attention.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            past_key_value: Cached key-value state from previous decoding steps
            output_attentions: Whether to return attention probabilities
            
        Returns:
            Attention output [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project inputs to queries, keys, and values
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape to [batch_size, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Combine with past key-value states if provided
        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=1)
            v = torch.cat([past_value, v], dim=1)
            seq_len_k = k.shape[1]
        else:
            seq_len_k = seq_len
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask to have the same shape as k
            # [batch_size, seq_len] -> [batch_size, seq_len, 1, 1]
            mask = attention_mask.unsqueeze(-1).unsqueeze(-1)
            
            # Apply mask to keys and values
            k = k * mask
            v = v * mask
        
        # Apply kernel feature maps if using dimension reduction
        if self.dim > 0:
            # Project to lower dimension: [batch_size, seq_len, num_heads, dim]
            q = self.q_map(q)
            k = self.k_map(k)
        
        # Apply kernel function to keys and queries
        q = self._apply_kernel_function(q)
        k = self._apply_kernel_function(k)
        
        # Handle causal masking for autoregressive models
        if self.causal:
            # For causal masking, we need to process sequence elements one by one
            # Initialize output and key-value cumulative sums
            output = torch.zeros_like(q)
            
            # Compute linear attention for each position in sequence
            cumulative_k = torch.zeros(batch_size, self.num_heads, 1, self.dim if self.dim > 0 else self.head_dim, device=q.device)
            cumulative_kv = torch.zeros(batch_size, self.num_heads, 1, self.head_dim, device=q.device)
            
            for i in range(seq_len_k):
                # Get key and value at current position
                k_i = k[:, i].unsqueeze(1)  # [batch_size, 1, num_heads, dim]
                v_i = v[:, i].unsqueeze(1)  # [batch_size, 1, num_heads, head_dim]
                
                # Update cumulative sums
                cumulative_k = cumulative_k + k_i.transpose(1, 2)
                cumulative_kv = cumulative_kv + torch.matmul(
                    k_i.transpose(1, 2), 
                    v_i.transpose(1, 2)
                )
                
                # Only update output if this is a valid position in q
                if i < seq_len:
                    # Get query at current position
                    q_i = q[:, i].unsqueeze(1).transpose(1, 2)  # [batch_size, num_heads, 1, dim]
                    
                    # Compute attention using cumulative sums
                    normalizer = torch.matmul(q_i, cumulative_k.transpose(2, 3)) + self.eps
                    output_i = torch.matmul(cumulative_kv, q_i.transpose(2, 3)) / normalizer
                    
                    # Update output
                    output[:, i] = output_i.squeeze(2)
        else:
            # Standard linear attention (non-causal)
            # Transpose dimensions for matrix multiplication
            # [batch_size, seq_len, num_heads, dim] -> [batch_size, num_heads, seq_len, dim]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # Linear attention computation: softmax(Q)(softmax(K)^T V)
            # First compute normalizer: [batch_size, num_heads, seq_len, seq_len_k]
            kv = torch.matmul(k.transpose(2, 3), v)
            
            # Then compute attention output: [batch_size, num_heads, seq_len, head_dim]
            normalizer = torch.matmul(q, k.sum(dim=2).unsqueeze(2)) + self.eps
            output = torch.matmul(q.transpose(2, 3), kv).transpose(2, 3) / normalizer
        
        # Transpose back and reshape: [batch_size, seq_len, hidden_size]
        output = output.transpose(1, 2).reshape(batch_size, -1, self.hidden_size)
        
        # Final projection
        output = self.out_proj(output)
        
        # Apply dropout
        output = self.dropout(output)
        
        if past_key_value is not None:
            return output, (k, v)
        
        return output


class MultiQueryLinearAttention(LinearAttention):
    """
    Multi-Query Linear Attention variant.
    Uses a single set of keys and values shared across all attention heads,
    but separate queries for each head.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        dim: int = 64,
        eps: float = 1e-6,
        causal: bool = False,
        bias: bool = True,
        kernel_function: str = "elu",
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            dim=dim,
            eps=eps,
            causal=causal,
            bias=bias,
            kernel_function=kernel_function,
        )
        
        # Override key and value projections for multi-query attention
        self.k_proj = nn.Linear(hidden_size, self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.head_dim, bias=bias)
        
        # Override feature maps for keys if using dimension reduction
        if dim > 0:
            self.k_map = nn.Linear(self.head_dim, dim, bias=False)
        
        # Re-initialize the modified parameters
        self._reset_parameters()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Apply multi-query linear attention.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            past_key_value: Cached key-value state from previous decoding steps
            output_attentions: Whether to return attention probabilities
            
        Returns:
            Attention output [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project inputs to queries, keys, and values
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape queries: [batch_size, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # For multi-query attention, keys and values are shared across heads
        # [batch_size, seq_len, head_dim]
        
        # Combine with past key-value states if provided
        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=1)
            v = torch.cat([past_value, v], dim=1)
            seq_len_k = k.shape[1]
        else:
            seq_len_k = seq_len
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask
            mask = attention_mask.unsqueeze(-1)
            
            # Apply mask to keys and values
            k = k * mask
            v = v * mask
        
        # Apply kernel feature maps if using dimension reduction
        if self.dim > 0:
            # Project to lower dimension
            q = self.q_map(q)
            k = self.k_map(k)
        
        # Apply kernel function
        q = self._apply_kernel_function(q)
        k = self._apply_kernel_function(k)
        
        # Transpose queries for matrix multiplication
        # [batch_size, num_heads, seq_len, dim]
        q = q.transpose(1, 2)
        
        if self.causal:
            # Initialize output
            output = torch.zeros_like(q)
            
            # Compute linear attention for each position
            cumulative_k = torch.zeros(batch_size, 1, self.dim if self.dim > 0 else self.head_dim, device=q.device)
            cumulative_kv = torch.zeros(batch_size, 1, self.head_dim, device=q.device)
            
            for i in range(seq_len_k):
                # Get key and value at current position
                k_i = k[:, i].unsqueeze(1)  # [batch_size, 1, dim]
                v_i = v[:, i].unsqueeze(1)  # [batch_size, 1, head_dim]
                
                # Update cumulative sums
                cumulative_k = cumulative_k + k_i
                cumulative_kv = cumulative_kv + k_i.transpose(1, 2) * v_i
                
                # Update output if valid position
                if i < seq_len:
                    # Get query at current position for all heads
                    q_i = q[:, :, i].unsqueeze(2)  # [batch_size, num_heads, 1, dim]
                    
                    # Compute attention using cumulative sums
                    normalizer = torch.matmul(q_i, cumulative_k.transpose(1, 2).unsqueeze(1)) + self.eps
                    output_i = torch.matmul(q_i, cumulative_kv.unsqueeze(1)) / normalizer
                    
                    # Update output
                    output[:, :, i] = output_i.squeeze(2)
        else:
            # Standard linear attention (non-causal) for multi-query
            # First compute key-value product: [batch_size, head_dim, head_dim]
            kv = torch.matmul(k.transpose(1, 2), v)
            
            # Compute normalizer: [batch_size, num_heads, seq_len, 1]
            k_sum = k.sum(dim=1).unsqueeze(1)  # [batch_size, 1, dim]
            normalizer = torch.matmul(q, k_sum.unsqueeze(1).transpose(2, 3)) + self.eps
            
            # Compute attention output: [batch_size, num_heads, seq_len, head_dim]
            output = torch.matmul(q, kv.unsqueeze(1)) / normalizer
        
        # Transpose back and reshape: [batch_size, seq_len, hidden_size]
        output = output.transpose(1, 2).reshape(batch_size, -1, self.hidden_size)
        
        # Final projection
        output = self.out_proj(output)
        
        # Apply dropout
        output = self.dropout(output)
        
        if past_key_value is not None:
            return output, (k, v)
        
        return output 