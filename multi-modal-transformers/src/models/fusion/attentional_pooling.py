import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionalPooling(nn.Module):
    """
    Attentional pooling mechanism to convert sequence representations
    into fixed-size vectors. This is more expressive than simple mean
    or max pooling, as it learns to weight different tokens based on their
    importance.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 1,
        dropout: float = 0.1,
    ):
        """
        Initialize attentional pooling module.

        Args:
            hidden_size: Dimension of input features
            num_heads: Number of attention heads for pooling
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Check if hidden size is divisible by number of heads
        assert (
            hidden_size % num_heads == 0
        ), "hidden_size must be divisible by num_heads"
        self.head_dim = hidden_size // num_heads

        # Query vector (learnable)
        self.query = nn.Parameter(torch.zeros(1, num_heads, self.head_dim))

        # Key and value projections
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        # Output projection (used when num_heads > 1)
        self.output_projection = nn.Linear(hidden_size, hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize the learnable parameters."""
        nn.init.normal_(self.query, mean=0, std=0.02)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.zeros_(self.key.bias)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.zeros_(self.value.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply attentional pooling to extract a fixed-size representation.

        Args:
            hidden_states: Input sequence [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Pooled representation [batch_size, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project inputs to keys and values
        # [batch_size, seq_len, hidden_size]
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        # Reshape for multi-head attention
        # [batch_size, seq_len, num_heads, head_dim]
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Expand query for each item in batch
        # [1, num_heads, head_dim] -> [batch_size, num_heads, 1, head_dim]
        q = self.query.expand(batch_size, -1, -1).unsqueeze(2)

        # Scale factor for attention
        scale = 1.0 / math.sqrt(self.head_dim)

        # Compute attention scores: [batch_size, num_heads, 1, seq_len]
        attention_scores = torch.matmul(q, k.transpose(2, 3)) * scale

        # Apply attention mask if provided
        if attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            mask = attention_mask.unsqueeze(1).unsqueeze(1)

            # Set masked positions to negative infinity
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        # [batch_size, num_heads, 1, seq_len]
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        # [batch_size, num_heads, 1, seq_len] x [batch_size, num_heads, seq_len, head_dim]
        # -> [batch_size, num_heads, 1, head_dim]
        context = torch.matmul(attention_weights, v)

        # Reshape: [batch_size, num_heads, 1, head_dim] -> [batch_size, 1, hidden_size]
        context = context.transpose(1, 2).reshape(batch_size, 1, self.hidden_size)

        # Final projection if using multiple heads
        if self.num_heads > 1:
            context = self.output_projection(context)

        # Remove sequence dimension
        # [batch_size, 1, hidden_size] -> [batch_size, hidden_size]
        pooled_output = context.squeeze(1)

        return pooled_output


class MultiQueryAttentionalPooling(nn.Module):
    """
    Multi-query attentional pooling that uses multiple learnable queries
    to capture different aspects of the input sequence.
    """

    def __init__(
        self,
        hidden_size: int,
        num_queries: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize multi-query attentional pooling module.

        Args:
            hidden_size: Dimension of input features
            num_queries: Number of learnable query vectors
            num_heads: Number of attention heads per query
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        self.num_heads = num_heads

        # Check if hidden size is divisible by number of heads
        assert (
            hidden_size % num_heads == 0
        ), "hidden_size must be divisible by num_heads"
        self.head_dim = hidden_size // num_heads

        # Multiple query vectors (learnable)
        self.queries = nn.Parameter(
            torch.zeros(1, num_queries, num_heads, self.head_dim)
        )

        # Key and value projections
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        # Output projection
        self.output_projection = nn.Linear(num_queries * hidden_size, hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize the learnable parameters."""
        nn.init.normal_(self.queries, mean=0, std=0.02)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.zeros_(self.key.bias)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.zeros_(self.value.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply multi-query attentional pooling to extract a fixed-size representation.

        Args:
            hidden_states: Input sequence [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Pooled representation [batch_size, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project inputs to keys and values
        # [batch_size, seq_len, hidden_size]
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        # Reshape for multi-head attention
        # [batch_size, seq_len, num_heads, head_dim]
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Expand queries for each item in batch
        # [1, num_queries, num_heads, head_dim] -> [batch_size, num_queries, num_heads, 1, head_dim]
        q = self.queries.expand(batch_size, -1, -1, -1).unsqueeze(3)

        # Scale factor for attention
        scale = 1.0 / math.sqrt(self.head_dim)

        # Initialize context tensor to collect outputs from all queries
        context_list = []

        # Process each query separately
        for query_idx in range(self.num_queries):
            # Select current query: [batch_size, 1, num_heads, 1, head_dim]
            current_q = q[:, query_idx : query_idx + 1]

            # Compute attention scores: [batch_size, 1, num_heads, 1, seq_len]
            attention_scores = (
                torch.matmul(current_q, k.unsqueeze(1).transpose(3, 4)) * scale
            )

            # Remove extra dimensions: [batch_size, num_heads, 1, seq_len]
            attention_scores = attention_scores.squeeze(1)

            # Apply attention mask if provided
            if attention_mask is not None:
                # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
                mask = attention_mask.unsqueeze(1).unsqueeze(1)

                # Set masked positions to negative infinity
                attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

            # Apply softmax to get attention weights
            # [batch_size, num_heads, 1, seq_len]
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)

            # Apply attention to values
            # [batch_size, num_heads, 1, seq_len] x [batch_size, num_heads, seq_len, head_dim]
            # -> [batch_size, num_heads, 1, head_dim]
            current_context = torch.matmul(attention_weights, v)

            # Reshape: [batch_size, num_heads, 1, head_dim] -> [batch_size, hidden_size]
            current_context = current_context.transpose(1, 2).reshape(
                batch_size, 1, self.hidden_size
            )

            # Store result
            context_list.append(current_context)

        # Concatenate all query results: [batch_size, num_queries, hidden_size]
        context = torch.cat(context_list, dim=1)

        # Flatten and project to final representation
        # [batch_size, num_queries, hidden_size] -> [batch_size, num_queries * hidden_size]
        context_flat = context.reshape(batch_size, -1)

        # Final projection to get pooled representation: [batch_size, hidden_size]
        pooled_output = self.output_projection(context_flat)

        return pooled_output


class HierarchicalAttentionalPooling(nn.Module):
    """
    Hierarchical attentional pooling that first pools local windows and then
    pools across the window representations. This is useful for very long sequences.
    """

    def __init__(
        self,
        hidden_size: int,
        window_size: int = 128,
        stride: int = 64,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize hierarchical attentional pooling module.

        Args:
            hidden_size: Dimension of input features
            window_size: Size of local attention windows
            stride: Stride between windows (overlap when < window_size)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.stride = stride
        self.num_heads = num_heads

        # Check if hidden size is divisible by number of heads
        assert (
            hidden_size % num_heads == 0
        ), "hidden_size must be divisible by num_heads"
        self.head_dim = hidden_size // num_heads

        # Window-level pooling
        self.window_pooling = AttentionalPooling(
            hidden_size=hidden_size, num_heads=num_heads, dropout=dropout
        )

        # Sequence-level pooling
        self.sequence_pooling = AttentionalPooling(
            hidden_size=hidden_size, num_heads=num_heads, dropout=dropout
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply hierarchical attentional pooling to extract a fixed-size representation.

        Args:
            hidden_states: Input sequence [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Pooled representation [batch_size, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # Handle short sequences directly
        if seq_len <= self.window_size:
            return self.sequence_pooling(hidden_states, attention_mask)

        # Create windows
        window_representations = []
        window_masks = [] if attention_mask is not None else None

        # Process sequence in windows
        for start_idx in range(0, seq_len, self.stride):
            end_idx = min(start_idx + self.window_size, seq_len)

            # Skip empty windows
            if start_idx >= seq_len:
                break

            # Extract window
            window = hidden_states[:, start_idx:end_idx]

            # Extract corresponding mask
            if attention_mask is not None:
                window_mask = attention_mask[:, start_idx:end_idx]
                window_masks.append(window_mask)
            else:
                window_mask = None

            # Pool window
            window_rep = self.window_pooling(window, window_mask)
            window_representations.append(window_rep)

        # Combine window representations
        # [batch_size, num_windows, hidden_size]
        windows_combined = torch.stack(window_representations, dim=1)

        # Create mask for combined windows if needed
        if window_masks is not None:
            # Check if each window has at least one non-masked token
            window_combined_mask = torch.stack(
                [mask.sum(dim=1) > 0 for mask in window_masks], dim=1
            ).float()
        else:
            window_combined_mask = None

        # Apply sequence-level pooling
        pooled_output = self.sequence_pooling(windows_combined, window_combined_mask)

        return pooled_output
