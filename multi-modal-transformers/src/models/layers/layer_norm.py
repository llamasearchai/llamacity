import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class LayerNorm(nn.Module):
    """
    Layer normalization module with optional bias.
    Normalizes across the feature dimension.
    """
    
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ):
        """
        Initialize layer normalization.
        
        Args:
            normalized_shape: Feature size to normalize over
            eps: Small value for numerical stability
            elementwise_affine: Whether to use learnable scale and shift
            bias: Whether to use a bias term
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape)) if bias else None
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization to input.
        
        Args:
            x: Input tensor [batch_size, seq_len, normalized_shape] or [batch_size, normalized_shape]
            
        Returns:
            Normalized tensor of same shape as input
        """
        # Calculate mean and variance along last dimension
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        
        # Apply elementwise affine transformation if specified
        if self.elementwise_affine:
            x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
        
        return x


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    From the paper "Root Mean Square Layer Normalization" (https://arxiv.org/abs/1910.07467)
    """
    
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ):
        """
        Initialize RMS normalization.
        
        Args:
            normalized_shape: Feature size to normalize over
            eps: Small value for numerical stability
            elementwise_affine: Whether to use learnable scale
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("weight", None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization to input.
        
        Args:
            x: Input tensor [batch_size, seq_len, normalized_shape] or [batch_size, normalized_shape]
            
        Returns:
            Normalized tensor of same shape as input
        """
        # Calculate RMS along last dimension
        # sqrt(1/n * sum(x_i^2)) for i from 1 to n
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        
        # Normalize
        x = x / rms
        
        # Apply elementwise affine transformation if specified
        if self.elementwise_affine:
            x = x * self.weight
        
        return x


class ScaleNorm(nn.Module):
    """
    Scale Normalization.
    From the paper "Transformers without Tears: Improving the Normalization of Self-Attention".
    Normalizes by the L2 norm of the entire layer.
    """
    
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        scale: Optional[float] = None,
    ):
        """
        Initialize scale normalization.
        
        Args:
            normalized_shape: Feature size (used for initialization only)
            eps: Small value for numerical stability
            scale: Initial scale factor (if None, use sqrt(normalized_shape))
        """
        super().__init__()
        self.eps = eps
        
        # Initialize scale parameter
        if scale is None:
            scale = math.sqrt(normalized_shape)
        self.scale = nn.Parameter(torch.tensor(scale))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply scale normalization to input.
        
        Args:
            x: Input tensor [batch_size, seq_len, normalized_shape] or [batch_size, normalized_shape]
            
        Returns:
            Normalized tensor of same shape as input
        """
        # Calculate L2 norm along last dimension
        norm = torch.norm(x, dim=-1, keepdim=True) + self.eps
        
        # Normalize using learnable scale
        x = self.scale * x / norm
        
        return x


class GroupNorm1D(nn.Module):
    """
    Group Normalization for 1D sequential data.
    Divides channels into groups and normalizes separately within each group.
    Adapted for NLP/transformer applications.
    """
    
    def __init__(
        self,
        num_channels: int,
        num_groups: int = 32,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        """
        Initialize group normalization.
        
        Args:
            num_channels: Number of channels (feature dimension)
            num_groups: Number of groups to separate channels into
            eps: Small value for numerical stability
            affine: Whether to use learnable scale and shift
        """
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        
        if num_channels % num_groups != 0:
            raise ValueError(f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})")
        
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply group normalization to input.
        
        Args:
            x: Input tensor [batch_size, seq_len, num_channels]
            
        Returns:
            Normalized tensor of same shape as input
        """
        # Move channel dimension to the right position for grouping
        # [batch_size, seq_len, num_channels] -> [batch_size, num_channels, seq_len]
        x = x.transpose(1, 2)
        
        # Get shape
        batch_size, num_channels, seq_len = x.shape
        
        # Reshape for grouping
        # [batch_size, num_channels, seq_len] -> [batch_size, num_groups, channels_per_group, seq_len]
        x = x.reshape(batch_size, self.num_groups, -1, seq_len)
        
        # Normalize over channels within each group
        # Calculate mean and variance along 2 (channels_per_group) and 3 (seq_len)
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        # Reshape back
        # [batch_size, num_groups, channels_per_group, seq_len] -> [batch_size, num_channels, seq_len]
        x = x.reshape(batch_size, num_channels, seq_len)
        
        # Apply affine transformation if specified
        if self.affine:
            # [batch_size, num_channels, seq_len]
            x = x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        
        # Return to original shape
        # [batch_size, num_channels, seq_len] -> [batch_size, seq_len, num_channels]
        x = x.transpose(1, 2)
        
        return x 