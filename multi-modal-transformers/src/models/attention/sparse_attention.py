import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import einops


class SparseMultiHeadAttention(nn.Module):
    """
    Sparse Multi-Head Attention with local windows and global tokens.
    This attention mechanism reduces computation for long sequences by:
    1. Computing attention only within local windows
    2. Allowing certain "global" tokens to attend to the entire sequence
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        window_size: int = 256,
        stride: int = 128,
        global_tokens: int = 1,
        bias: bool = True,
        causal: bool = False,
    ):
        """
        Initialize sparse attention module.
        
        Args:
            hidden_size: Dimensionality of embedding vectors
            num_heads: Number of attention heads
            dropout: Dropout probability
            window_size: Size of local attention windows
            stride: Stride between windows (overlap when < window_size)
            global_tokens: Number of initial tokens that attend to entire sequence
            bias: Whether to use bias in projections
            causal: Whether to use causal masking (for autoregressive models)
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.stride = stride
        self.global_tokens = global_tokens
        self.causal = causal
        
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        # Projection matrices
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
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
    
    def _compute_windows(self, seq_len: int) -> torch.Tensor:
        """
        Compute the indices for windows.
        
        Args:
            seq_len: Length of the sequence
            
        Returns:
            Tensor of indices for each window [num_windows, window_size]
        """
        # Calculate number of windows
        num_windows = max(1, (seq_len - self.window_size) // self.stride + 1)
        
        # Create indices for each window
        indices = []
        for i in range(num_windows):
            start_idx = i * self.stride
            end_idx = min(start_idx + self.window_size, seq_len)
            window_indices = torch.arange(start_idx, end_idx)
            
            # Pad if necessary to maintain constant window size
            if len(window_indices) < self.window_size:
                padding = torch.arange(end_idx - 1, end_idx - 1 - (self.window_size - len(window_indices)), -1)
                window_indices = torch.cat([window_indices, padding])
            
            indices.append(window_indices)
        
        # Stack all window indices
        return torch.stack(indices)
    
    def _global_local_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply global-local attention.
        Global tokens attend to the entire sequence.
        Local tokens attend only within their window.
        
        Args:
            q: Query tensor [batch_size, seq_len, num_heads, head_dim]
            k: Key tensor [batch_size, seq_len, num_heads, head_dim]
            v: Value tensor [batch_size, seq_len, num_heads, head_dim]
            attention_mask: Mask [batch_size, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, num_heads, head_dim = q.shape
        device = q.device
        
        # Handle special case for short sequences
        if seq_len <= self.window_size + self.global_tokens:
            # Switch to full attention for short sequences
            q_full = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
            k_full = k.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
            v_full = v.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
            
            # Compute attention scores
            attn_weights = torch.matmul(q_full, k_full.transpose(2, 3)) * self.scale
            
            # Apply causal mask if needed
            if self.causal:
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=device) * float("-inf"), diagonal=1
                )
                attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)
            
            # Apply attention mask if provided
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
                attn_weights = attn_weights.masked_fill(attention_mask == 0, float("-inf"))
            
            # Apply softmax and dropout
            attn_probs = F.softmax(attn_weights, dim=-1)
            attn_probs = self.dropout(attn_probs)
            
            # Apply attention to values
            context = torch.matmul(attn_probs, v_full)  # [batch_size, num_heads, seq_len, head_dim]
            
            # Transpose and reshape back
            output = context.transpose(1, 2).reshape(batch_size, seq_len, -1)
            return output
        
        # Split into global and local tokens
        q_global = q[:, :self.global_tokens]
        k_global = k[:, :self.global_tokens]
        v_global = v[:, :self.global_tokens]
        
        q_local = q[:, self.global_tokens:]
        k_local = k[:, self.global_tokens:]
        v_local = v[:, self.global_tokens:]
        
        # Process global tokens with full attention
        q_global = q_global.transpose(1, 2)  # [batch_size, num_heads, global_tokens, head_dim]
        k_t = k.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        v_t = v.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Compute global attention scores: [batch_size, num_heads, global_tokens, seq_len]
        global_attn_weights = torch.matmul(q_global, k_t.transpose(2, 3)) * self.scale
        
        # Apply causal mask for global tokens if needed
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(self.global_tokens, seq_len, device=device) * float("-inf"), diagonal=1
            )
            global_attn_weights = global_attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply attention mask for global tokens if provided
        if attention_mask is not None:
            global_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            global_attn_weights = global_attn_weights.masked_fill(global_mask == 0, float("-inf"))
        
        # Apply softmax and dropout for global attention
        global_attn_probs = F.softmax(global_attn_weights, dim=-1)
        global_attn_probs = self.dropout(global_attn_probs)
        
        # Apply global attention to values
        global_context = torch.matmul(global_attn_probs, v_t)  # [batch_size, num_heads, global_tokens, head_dim]
        
        # Compute window indices for local attention
        local_seq_len = seq_len - self.global_tokens
        window_indices = self._compute_windows(local_seq_len).to(device)
        num_windows = window_indices.shape[0]
        
        # Initialize output tensor for local tokens
        local_output = torch.zeros(
            batch_size, local_seq_len, self.hidden_size, device=device
        )
        
        # Process each window separately to save memory
        for i in range(num_windows):
            window_idx = window_indices[i]
            
            # Get queries for current window's centers
            start_idx = i * self.stride
            end_idx = min(start_idx + self.window_size, local_seq_len)
            window_q_indices = torch.arange(start_idx, end_idx, device=device)
            
            # Skip empty windows
            if len(window_q_indices) == 0:
                continue
            
            window_q = q_local[:, window_q_indices]  # [batch_size, window_q_len, num_heads, head_dim]
            window_q = window_q.transpose(1, 2)  # [batch_size, num_heads, window_q_len, head_dim]
            
            # Get keys and values for the window
            window_k = k_local[:, window_idx]  # [batch_size, window_size, num_heads, head_dim]
            window_v = v_local[:, window_idx]  # [batch_size, window_size, num_heads, head_dim]
            
            window_k = window_k.transpose(1, 2)  # [batch_size, num_heads, window_size, head_dim]
            window_v = window_v.transpose(1, 2)  # [batch_size, num_heads, window_size, head_dim]
            
            # Compute window attention scores
            window_attn_weights = torch.matmul(window_q, window_k.transpose(2, 3)) * self.scale
            
            # Apply causal mask within window if needed
            if self.causal:
                # Create relative position mask within the window
                q_pos = torch.arange(len(window_q_indices), device=device).unsqueeze(1)
                k_pos = torch.arange(len(window_idx), device=device).unsqueeze(0)
                rel_pos = window_idx.unsqueeze(0) - (self.global_tokens + window_q_indices).unsqueeze(1)
                causal_mask = (rel_pos < 0).float() * float("-inf")
                window_attn_weights = window_attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)
            
            # Apply attention mask within window if provided
            if attention_mask is not None:
                # Get window portion of the mask
                window_mask = attention_mask[:, self.global_tokens + window_idx]
                window_mask = window_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, window_size]
                window_attn_weights = window_attn_weights.masked_fill(window_mask == 0, float("-inf"))
            
            # Apply softmax and dropout for window attention
            window_attn_probs = F.softmax(window_attn_weights, dim=-1)
            window_attn_probs = self.dropout(window_attn_probs)
            
            # Apply window attention to values
            window_context = torch.matmul(window_attn_probs, window_v)  # [batch_size, num_heads, window_q_len, head_dim]
            
            # Transpose and reshape back
            window_output = window_context.transpose(1, 2).reshape(batch_size, len(window_q_indices), -1)
            
            # Add to output tensor
            local_output[:, window_q_indices] += window_output
        
        # Normalize the overlapped regions (divide by number of contributions)
        overlap_counts = torch.zeros(local_seq_len, device=device)
        for i in range(num_windows):
            start_idx = i * self.stride
            end_idx = min(start_idx + self.window_size, local_seq_len)
            overlap_counts[start_idx:end_idx] += 1
        
        # Handle potential zero counts (although shouldn't happen with proper windows)
        overlap_counts = torch.clamp(overlap_counts, min=1.0)
        overlap_counts = overlap_counts.unsqueeze(0).unsqueeze(2)  # [1, local_seq_len, 1]
        local_output = local_output / overlap_counts
        
        # Combine global and local outputs
        global_output = global_context.transpose(1, 2).reshape(batch_size, self.global_tokens, -1)
        output = torch.cat([global_output, local_output], dim=1)
        
        return output
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Apply sparse multi-head attention.
        
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
            seq_len = k.shape[1]
        
        # Apply global-local attention mechanism
        context = self._global_local_attention(q, k, v, attention_mask)
        
        # Final projection
        output = self.out_proj(context)
        
        # Return key-value states for potential future use
        if past_key_value is not None:
            new_key_value = (k, v)
            return output, new_key_value
        
        return output 