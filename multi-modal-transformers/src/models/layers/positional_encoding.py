import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Positional encoding using sine and cosine functions.

    This adds positional information to the input embeddings
    using the approach from "Attention Is All You Need" paper.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        learnable: bool = False,
    ):
        """
        Initialize positional encoding.

        Args:
            d_model: Dimension of embedding vectors
            max_len: Maximum sequence length to support
            dropout: Dropout probability
            learnable: Whether to use learnable positional embeddings
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len
        self.learnable = learnable

        if learnable:
            # Learnable positional embeddings
            self.position_embeddings = nn.Parameter(torch.zeros(1, max_len, d_model))
            self._reset_parameters()
        else:
            # Fixed positional encodings
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )

            # Apply sine to even indices and cosine to odd indices
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            # Add batch dimension and register as buffer (not a parameter)
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)

    def _reset_parameters(self):
        """Initialize learnable parameters if using learnable positional embeddings."""
        if self.learnable:
            # Initialize with sinusoidal pattern for better convergence
            position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2).float()
                * (-math.log(10000.0) / self.d_model)
            )

            pe = torch.zeros(self.max_len, self.d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            # Set parameter to pre-computed values
            with torch.no_grad():
                self.position_embeddings.copy_(pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input embeddings [batch_size, seq_len, d_model]

        Returns:
            Embeddings with positional information
        """
        if self.learnable:
            # Use learnable positional embeddings
            seq_len = x.size(1)
            x = x + self.position_embeddings[:, :seq_len, :]
        else:
            # Add fixed positional encodings
            seq_len = x.size(1)
            x = x + self.pe[:, :seq_len, :]

        return self.dropout(x)


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding (RoPE) from "Roformer: Enhanced Transformer with Rotary Position Embedding".

    This applies rotation to the input embeddings to encode positional information.
    Particularly effective for relative positional encoding.
    """

    def __init__(
        self,
        dim: int,
        max_len: int = 5000,
        base: int = 10000,
    ):
        """
        Initialize rotary positional encoding.

        Args:
            dim: Feature dimension
            max_len: Maximum sequence length
            base: Base for the rotation calculation
        """
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base

        # Compute the rotation matrices in advance
        self.create_rotary_matrix(device=None)

    def create_rotary_matrix(self, device: Optional[torch.device] = None):
        """
        Create rotation matrices for all positions up to max_len.

        Args:
            device: Device to create the matrices on
        """
        # Set half dimension (we will rotate pairs of dimensions)
        half_dim = self.dim // 2

        # Create position indices: [max_len]
        positions = torch.arange(self.max_len, device=device)

        # Create dimension indices: [half_dim]
        dims = torch.arange(half_dim, device=device)

        # Compute frequency for each dimension: [half_dim]
        freqs = 1.0 / (self.base ** (dims / half_dim))

        # Create position-frequency matrix: [max_len, half_dim]
        t = positions.unsqueeze(1) * freqs.unsqueeze(0)

        # Compute rotation matrix components
        # [max_len, half_dim]
        freqs_cos = torch.cos(t)
        freqs_sin = torch.sin(t)

        # Register as buffers
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half of the features by swapping and negating them.

        Args:
            x: Input tensor

        Returns:
            Rotated tensor
        """
        # Split embedding dimension in half
        x1, x2 = x.chunk(2, dim=-1)

        # Concatenate with negative of second half first
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional encoding to input embeddings.

        Args:
            x: Input embeddings [batch_size, seq_len, dim]

        Returns:
            Embeddings with rotary positional encoding
        """
        seq_len = x.size(1)

        # Ensure we have computed rotation matrices on the correct device
        if self.freqs_cos.device != x.device:
            self.create_rotary_matrix(x.device)

        # Get rotation matrices for the current sequence
        freqs_cos = self.freqs_cos[:seq_len, :]
        freqs_sin = self.freqs_sin[:seq_len, :]

        # Reshape for broadcasting
        # [seq_len, dim/2] -> [1, seq_len, 1, dim/2]
        freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(2)
        freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(2)

        # Reshape input to separate heads if needed
        if len(x.shape) == 4:
            # [batch_size, seq_len, num_heads, head_dim]
            batch, seq, heads, dims = x.shape

            # Ensure dim is dividable by 2
            assert dims % 2 == 0, "Feature dimension must be divisible by 2"

            # Expand frequency shapes for broadcasting with multiple heads
            freqs_cos = freqs_cos.expand(batch, seq, heads, dims // 2)
            freqs_sin = freqs_sin.expand(batch, seq, heads, dims // 2)
        else:
            # [batch_size, seq_len, dim]
            batch, seq, dims = x.shape

            # Ensure dim is dividable by 2
            assert dims % 2 == 0, "Feature dimension must be divisible by 2"

            # Expand frequency shapes for broadcasting
            freqs_cos = freqs_cos.squeeze(2).expand(batch, seq, dims // 2)
            freqs_sin = freqs_sin.squeeze(2).expand(batch, seq, dims // 2)

        # Split embedding dimension in half
        x_half = x.chunk(2, dim=-1)[0]

        # Apply rotation using the rotated half
        x_rotated = self._rotate_half(x)

        # First half is multiplied by cos, rotated half by sin
        x_half_cos = x_half * freqs_cos
        x_half_sin = x_rotated.chunk(2, dim=-1)[0] * freqs_sin

        # Concatenate both parts for the final result
        return torch.cat((x_half_cos - x_half_sin, x_half_cos + x_half_sin), dim=-1)


class RelativePositionalEncoding(nn.Module):
    """
    Relative Positional Encoding for transformers.

    Keeps track of relative positions between tokens rather than absolute positions.
    Based on "Self-Attention with Relative Position Representations".
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        """
        Initialize relative positional encoding.

        Args:
            d_model: Dimension of embedding vectors
            max_len: Maximum sequence length to support
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len * 2 - 1  # Maximum possible relative distance

        # Create learnable embeddings for relative positions
        self.rel_embeddings = nn.Parameter(torch.zeros(self.max_len, d_model))

        # Initialize with a sinusoidal pattern
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize the relative position embeddings."""
        # Center positions at 0
        positions = torch.arange(
            -(self.max_len // 2), self.max_len // 2 + 1, dtype=torch.float
        )

        # Initialize with sinusoidal pattern
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * (-math.log(10000.0) / self.d_model)
        )

        # Apply sine to even indices and cosine to odd indices
        self.rel_embeddings.data[:, 0::2] = torch.sin(positions.unsqueeze(1) * div_term)
        self.rel_embeddings.data[:, 1::2] = torch.cos(positions.unsqueeze(1) * div_term)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create relative position indices for a given sequence length.

        Args:
            x: Input embeddings [batch_size, seq_len, d_model]

        Returns:
            Tuple of (embeddings, relative position matrix)
        """
        # This doesn't modify the input directly but returns the relative position matrix
        # to be used in the attention mechanism
        seq_len = x.size(1)

        # Create indices for all possible pairs of positions
        # [seq_len, seq_len]
        pos_indices = torch.arange(seq_len, device=x.device)
        rel_indices = pos_indices.unsqueeze(1) - pos_indices.unsqueeze(0)

        # Shift indices to be non-negative
        rel_indices = rel_indices + self.max_len // 2

        # Get the embeddings for each relative position
        # [seq_len, seq_len, d_model]
        rel_pos_embeddings = self.rel_embeddings[rel_indices]

        # Return the input embeddings and relative position matrix
        return self.dropout(x), rel_pos_embeddings
