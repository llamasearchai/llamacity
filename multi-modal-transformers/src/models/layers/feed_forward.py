import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable


class FeedForward(nn.Module):
    """
    Feed-forward network used in transformer models.
    Typically consists of two linear transformations with a non-linearity in between.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        bias: bool = True,
        activation_args: Optional[dict] = None,
    ):
        """
        Initialize feed-forward network.
        
        Args:
            d_model: Input and output dimension
            d_ff: Hidden dimension
            activation: Activation function ("relu", "gelu", "swish", "silu")
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
            activation_args: Additional arguments for activation function
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation_name = activation
        self.activation_args = activation_args or {}
        
        # Linear layers
        self.fc1 = nn.Linear(d_model, d_ff, bias=bias)
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Set up activation function
        self.activation = self._get_activation_fn(activation)
    
    def _get_activation_fn(self, activation: str) -> Callable:
        """
        Get activation function by name.
        
        Args:
            activation: Name of activation function
            
        Returns:
            Activation function
        """
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        elif activation == "swish" or activation == "silu":
            return F.silu
        elif activation == "mish":
            return F.mish
        elif activation == "leaky_relu":
            return lambda x: F.leaky_relu(x, negative_slope=self.activation_args.get("negative_slope", 0.1))
        else:
            raise ValueError(f"Unknown activation function: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward network to input.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # First linear layer and activation
        h = self.fc1(x)
        h = self.activation(h)
        
        # Dropout after activation
        h = self.dropout(h)
        
        # Second linear layer
        out = self.fc2(h)
        
        return out


class GLUFeedForward(nn.Module):
    """
    Gated Linear Unit (GLU) feed-forward network.
    Uses multiplicative gating for improved gradient flow.
    From the paper "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202)
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = "gelu",
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """
        Initialize GLU feed-forward network.
        
        Args:
            d_model: Input and output dimension
            d_ff: Hidden dimension (will be doubled internally for gating)
            activation: Activation function for gating
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation_name = activation
        
        # Linear layers
        self.fc1_1 = nn.Linear(d_model, d_ff, bias=bias)  # Value path
        self.fc1_2 = nn.Linear(d_model, d_ff, bias=bias)  # Gate path
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Set up activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "swish" or activation == "silu":
            self.activation = F.silu
        else:
            raise ValueError(f"Unknown activation function: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GLU feed-forward network to input.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Value path
        v = self.fc1_1(x)
        
        # Gate path with activation
        g = self.fc1_2(x)
        g = self.activation(g)
        
        # Element-wise multiplication of value and gate
        h = v * g
        
        # Dropout after gating
        h = self.dropout(h)
        
        # Final linear projection
        out = self.fc2(h)
        
        return out


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Feed-forward network used in PaLM and other recent models.
    Combines SwiSH activation and Gated Linear Unit for improved performance.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """
        Initialize SwiGLU feed-forward network.
        
        Args:
            d_model: Input and output dimension
            d_ff: Hidden dimension
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Linear layers
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_model, d_ff, bias=bias)
        self.w3 = nn.Linear(d_ff, d_model, bias=bias)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU feed-forward network to input.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # SwiGLU computation: SwiSH(x@w1) * (x@w2)
        swish = self.w1(x) * torch.sigmoid(self.w1(x) * 1.0)
        gate = self.w2(x)
        h = swish * gate
        
        # Dropout after activation
        h = self.dropout(h)
        
        # Final linear projection
        out = self.w3(h)
        
        return out 