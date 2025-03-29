import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
import logging
from dataclasses import dataclass

from .attention import MultiHeadCrossAttention, SparseMultiHeadAttention, LinearAttention
from .encoders import TextEncoder, ImageEncoder, AudioEncoder
from .fusion import CrossModalFusion, AttentionalPooling
from .layers import PositionalEncoding, FeedForward, LayerNorm

logger = logging.getLogger(__name__)

@dataclass
class MultiModalTransformerConfig:
    """Configuration class for MultiModalTransformer."""
    
    # General parameters
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: int = 0.1
    attention_probs_dropout_prob: int = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    use_cache: bool = True
    vocab_size: int = 30522
    
    # Text-specific parameters
    max_text_length: int = 512
    text_hidden_size: int = 768
    text_num_hidden_layers: int = 12
    text_num_attention_heads: int = 12
    
    # Image-specific parameters
    image_size: int = 224
    image_channels: int = 3
    image_patch_size: int = 16
    image_hidden_size: int = 768
    image_num_hidden_layers: int = 12
    image_num_attention_heads: int = 12
    
    # Audio-specific parameters
    audio_feature_size: int = 128
    audio_sequence_length: int = 1024
    audio_hidden_size: int = 768
    audio_num_hidden_layers: int = 12
    audio_num_attention_heads: int = 12
    
    # Fusion parameters
    fusion_type: str = "cross_attention"  # Options: cross_attention, concat, sum, weighted
    fusion_layers: int = 2
    use_modality_tokens: bool = True
    
    # Attention mechanisms
    attention_type: str = "standard"  # Options: standard, sparse, linear, performer
    sparse_attention_window: int = 256
    sparse_attention_stride: int = 128
    linear_attention_dim: int = 256
    
    # Training parameters
    gradient_checkpointing: bool = False


class MultiModalTransformerLayer(nn.Module):
    """A single transformer layer for multi-modal processing."""
    
    def __init__(self, config: MultiModalTransformerConfig):
        super().__init__()
        self.config = config
        
        # Self-attention layers for each modality
        if config.attention_type == "standard":
            self.text_self_attention = nn.MultiheadAttention(
                config.text_hidden_size, 
                config.text_num_attention_heads, 
                dropout=config.attention_probs_dropout_prob,
                batch_first=True
            )
            self.image_self_attention = nn.MultiheadAttention(
                config.image_hidden_size, 
                config.image_num_attention_heads, 
                dropout=config.attention_probs_dropout_prob,
                batch_first=True
            )
            self.audio_self_attention = nn.MultiheadAttention(
                config.audio_hidden_size, 
                config.audio_num_attention_heads, 
                dropout=config.attention_probs_dropout_prob,
                batch_first=True
            )
        elif config.attention_type == "sparse":
            self.text_self_attention = SparseMultiHeadAttention(
                config.text_hidden_size, 
                config.text_num_attention_heads,
                window_size=config.sparse_attention_window,
                stride=config.sparse_attention_stride
            )
            self.image_self_attention = SparseMultiHeadAttention(
                config.image_hidden_size, 
                config.image_num_attention_heads,
                window_size=config.sparse_attention_window,
                stride=config.sparse_attention_stride
            )
            self.audio_self_attention = SparseMultiHeadAttention(
                config.audio_hidden_size, 
                config.audio_num_attention_heads,
                window_size=config.sparse_attention_window,
                stride=config.sparse_attention_stride
            )
        elif config.attention_type == "linear":
            self.text_self_attention = LinearAttention(
                config.text_hidden_size, 
                config.text_num_attention_heads,
                dim=config.linear_attention_dim
            )
            self.image_self_attention = LinearAttention(
                config.image_hidden_size, 
                config.image_num_attention_heads,
                dim=config.linear_attention_dim
            )
            self.audio_self_attention = LinearAttention(
                config.audio_hidden_size, 
                config.audio_num_attention_heads,
                dim=config.linear_attention_dim
            )
        
        # Cross-attention layers between modalities
        self.text_image_cross_attn = MultiHeadCrossAttention(
            query_dim=config.text_hidden_size,
            key_dim=config.image_hidden_size,
            num_heads=config.text_num_attention_heads,
            dropout=config.attention_probs_dropout_prob
        )
        
        self.text_audio_cross_attn = MultiHeadCrossAttention(
            query_dim=config.text_hidden_size,
            key_dim=config.audio_hidden_size,
            num_heads=config.text_num_attention_heads,
            dropout=config.attention_probs_dropout_prob
        )
        
        self.image_text_cross_attn = MultiHeadCrossAttention(
            query_dim=config.image_hidden_size,
            key_dim=config.text_hidden_size,
            num_heads=config.image_num_attention_heads,
            dropout=config.attention_probs_dropout_prob
        )
        
        self.image_audio_cross_attn = MultiHeadCrossAttention(
            query_dim=config.image_hidden_size,
            key_dim=config.audio_hidden_size,
            num_heads=config.image_num_attention_heads,
            dropout=config.attention_probs_dropout_prob
        )
        
        self.audio_text_cross_attn = MultiHeadCrossAttention(
            query_dim=config.audio_hidden_size,
            key_dim=config.text_hidden_size,
            num_heads=config.audio_num_attention_heads,
            dropout=config.attention_probs_dropout_prob
        )
        
        self.audio_image_cross_attn = MultiHeadCrossAttention(
            query_dim=config.audio_hidden_size,
            key_dim=config.image_hidden_size,
            num_heads=config.audio_num_attention_heads,
            dropout=config.attention_probs_dropout_prob
        )
        
        # Feed-forward networks
        self.text_feed_forward = FeedForward(
            config.text_hidden_size, 
            config.intermediate_size, 
            dropout=config.hidden_dropout_prob
        )
        
        self.image_feed_forward = FeedForward(
            config.image_hidden_size, 
            config.intermediate_size, 
            dropout=config.hidden_dropout_prob
        )
        
        self.audio_feed_forward = FeedForward(
            config.audio_hidden_size, 
            config.intermediate_size, 
            dropout=config.hidden_dropout_prob
        )
        
        # Layer normalization
        self.text_layer_norm1 = LayerNorm(config.text_hidden_size, eps=config.layer_norm_eps)
        self.text_layer_norm2 = LayerNorm(config.text_hidden_size, eps=config.layer_norm_eps)
        self.text_layer_norm3 = LayerNorm(config.text_hidden_size, eps=config.layer_norm_eps)
        
        self.image_layer_norm1 = LayerNorm(config.image_hidden_size, eps=config.layer_norm_eps)
        self.image_layer_norm2 = LayerNorm(config.image_hidden_size, eps=config.layer_norm_eps)
        self.image_layer_norm3 = LayerNorm(config.image_hidden_size, eps=config.layer_norm_eps)
        
        self.audio_layer_norm1 = LayerNorm(config.audio_hidden_size, eps=config.layer_norm_eps)
        self.audio_layer_norm2 = LayerNorm(config.audio_hidden_size, eps=config.layer_norm_eps)
        self.audio_layer_norm3 = LayerNorm(config.audio_hidden_size, eps=config.layer_norm_eps)
        
        # Dropouts
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        audio_features: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the multi-modal transformer layer.
        
        Args:
            text_features: Text embeddings [batch_size, text_seq_len, hidden_size]
            image_features: Image embeddings [batch_size, image_seq_len, hidden_size]
            audio_features: Audio embeddings [batch_size, audio_seq_len, hidden_size]
            text_attention_mask: Mask for text self-attention
            image_attention_mask: Mask for image self-attention
            audio_attention_mask: Mask for audio self-attention
            
        Returns:
            Tuple of updated text, image, and audio features
        """
        # Process text modality
        residual = text_features
        
        # Self-attention
        if self.config.attention_type == "standard":
            text_features = self.text_layer_norm1(text_features)
            text_features, _ = self.text_self_attention(
                text_features, text_features, text_features,
                key_padding_mask=~text_attention_mask.bool() if text_attention_mask is not None else None
            )
        else:
            text_features = self.text_layer_norm1(text_features)
            text_features = self.text_self_attention(
                text_features,
                attention_mask=text_attention_mask
            )
            
        text_features = self.dropout(text_features)
        text_features = residual + text_features
        
        # Cross-attention with image
        residual = text_features
        text_features = self.text_layer_norm2(text_features)
        text_img_features = self.text_image_cross_attn(
            query=text_features,
            key=image_features,
            value=image_features,
            key_mask=image_attention_mask
        )
        
        # Cross-attention with audio
        text_audio_features = self.text_audio_cross_attn(
            query=text_features,
            key=audio_features,
            value=audio_features,
            key_mask=audio_attention_mask
        )
        
        # Combine cross-attention results
        text_cross_features = text_img_features + text_audio_features
        text_cross_features = self.dropout(text_cross_features)
        text_features = residual + text_cross_features
        
        # Feed-forward
        residual = text_features
        text_features = self.text_layer_norm3(text_features)
        text_features = self.text_feed_forward(text_features)
        text_features = self.dropout(text_features)
        text_features = residual + text_features
        
        # Process image modality (similar to text)
        residual = image_features
        
        # Self-attention
        if self.config.attention_type == "standard":
            image_features = self.image_layer_norm1(image_features)
            image_features, _ = self.image_self_attention(
                image_features, image_features, image_features,
                key_padding_mask=~image_attention_mask.bool() if image_attention_mask is not None else None
            )
        else:
            image_features = self.image_layer_norm1(image_features)
            image_features = self.image_self_attention(
                image_features,
                attention_mask=image_attention_mask
            )
            
        image_features = self.dropout(image_features)
        image_features = residual + image_features
        
        # Cross-attention with text and audio
        residual = image_features
        image_features = self.image_layer_norm2(image_features)
        
        img_text_features = self.image_text_cross_attn(
            query=image_features,
            key=text_features,
            value=text_features,
            key_mask=text_attention_mask
        )
        
        img_audio_features = self.image_audio_cross_attn(
            query=image_features,
            key=audio_features,
            value=audio_features,
            key_mask=audio_attention_mask
        )
        
        # Combine cross-attention results
        img_cross_features = img_text_features + img_audio_features
        img_cross_features = self.dropout(img_cross_features)
        image_features = residual + img_cross_features
        
        # Feed-forward
        residual = image_features
        image_features = self.image_layer_norm3(image_features)
        image_features = self.image_feed_forward(image_features)
        image_features = self.dropout(image_features)
        image_features = residual + image_features
        
        # Process audio modality (similar to text and image)
        residual = audio_features
        
        # Self-attention
        if self.config.attention_type == "standard":
            audio_features = self.audio_layer_norm1(audio_features)
            audio_features, _ = self.audio_self_attention(
                audio_features, audio_features, audio_features,
                key_padding_mask=~audio_attention_mask.bool() if audio_attention_mask is not None else None
            )
        else:
            audio_features = self.audio_layer_norm1(audio_features)
            audio_features = self.audio_self_attention(
                audio_features,
                attention_mask=audio_attention_mask
            )
            
        audio_features = self.dropout(audio_features)
        audio_features = residual + audio_features
        
        # Cross-attention with text and image
        residual = audio_features
        audio_features = self.audio_layer_norm2(audio_features)
        
        audio_text_features = self.audio_text_cross_attn(
            query=audio_features,
            key=text_features,
            value=text_features,
            key_mask=text_attention_mask
        )
        
        audio_img_features = self.audio_image_cross_attn(
            query=audio_features,
            key=image_features,
            value=image_features,
            key_mask=image_attention_mask
        )
        
        # Combine cross-attention results
        audio_cross_features = audio_text_features + audio_img_features
        audio_cross_features = self.dropout(audio_cross_features)
        audio_features = residual + audio_cross_features
        
        # Feed-forward
        residual = audio_features
        audio_features = self.audio_layer_norm3(audio_features)
        audio_features = self.audio_feed_forward(audio_features)
        audio_features = self.dropout(audio_features)
        audio_features = residual + audio_features
        
        return text_features, image_features, audio_features


class MultiModalTransformer(nn.Module):
    """
    Multi-modal transformer model with custom attention mechanisms.
    Supports text, image, and audio modalities.
    """
    
    def __init__(self, config: MultiModalTransformerConfig):
        super().__init__()
        self.config = config
        
        # Modality-specific encoders
        self.text_encoder = TextEncoder(config)
        self.image_encoder = ImageEncoder(config)
        self.audio_encoder = AudioEncoder(config)
        
        # Positional encodings
        self.text_position_embeddings = PositionalEncoding(
            config.text_hidden_size, 
            max_len=config.max_text_length
        )
        
        self.image_position_embeddings = PositionalEncoding(
            config.image_hidden_size, 
            max_len=(config.image_size // config.image_patch_size) ** 2 + 1  # +1 for CLS token
        )
        
        self.audio_position_embeddings = PositionalEncoding(
            config.audio_hidden_size, 
            max_len=config.audio_sequence_length
        )
        
        # Modality tokens (learnable type embeddings)
        if config.use_modality_tokens:
            self.modality_tokens = nn.Parameter(
                torch.randn(3, config.hidden_size)
            )
        
        # Multi-modal transformer layers
        self.layers = nn.ModuleList([
            MultiModalTransformerLayer(config)
            for _ in range(config.num_hidden_layers)
        ])
        
        # Cross-modal fusion
        self.fusion = CrossModalFusion(config)
        
        # Final attentional pooling
        self.pooling = AttentionalPooling(config.hidden_size)
        
        # Apply initialization
        self.apply(self._init_weights)
        
        logger.info(f"Initialized MultiModalTransformer with config: {config}")
        
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        text: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the multi-modal transformer.
        
        Args:
            text: Text input tokens [batch_size, text_seq_len]
            image: Image input [batch_size, channels, height, width]
            audio: Audio input [batch_size, audio_seq_len, audio_feature_size]
            text_attention_mask: Mask for text inputs
            image_attention_mask: Mask for image inputs
            audio_attention_mask: Mask for audio inputs
            return_dict: Whether to return outputs as a dict
            
        Returns:
            Dict containing modality outputs and fused representations
        """
        batch_size = self._get_batch_size(text, image, audio)
        device = self._get_device(text, image, audio)
        
        # Process text modality
        if text is not None:
            text_features = self.text_encoder(text)
            text_features = self.text_position_embeddings(text_features)
            
            # Add modality token if configured
            if self.config.use_modality_tokens:
                modality_token = self.modality_tokens[0].unsqueeze(0).expand(batch_size, 1, -1)
                text_features = torch.cat([modality_token, text_features], dim=1)
                
                # Update attention mask if present
                if text_attention_mask is not None:
                    modality_token_mask = torch.ones(batch_size, 1, device=device)
                    text_attention_mask = torch.cat([modality_token_mask, text_attention_mask], dim=1)
        else:
            # Create empty tensor with proper dimensions for absent modality
            text_features = torch.zeros(
                batch_size, 
                1,  # Just the modality token
                self.config.text_hidden_size, 
                device=device
            )
            text_attention_mask = torch.zeros(batch_size, 1, device=device)
        
        # Process image modality
        if image is not None:
            image_features = self.image_encoder(image)
            image_features = self.image_position_embeddings(image_features)
            
            # Add modality token if configured
            if self.config.use_modality_tokens:
                modality_token = self.modality_tokens[1].unsqueeze(0).expand(batch_size, 1, -1)
                image_features = torch.cat([modality_token, image_features], dim=1)
                
                # Update attention mask if present
                if image_attention_mask is not None:
                    modality_token_mask = torch.ones(batch_size, 1, device=device)
                    image_attention_mask = torch.cat([modality_token_mask, image_attention_mask], dim=1)
        else:
            # Create empty tensor with proper dimensions for absent modality
            image_features = torch.zeros(
                batch_size, 
                1,  # Just the modality token
                self.config.image_hidden_size, 
                device=device
            )
            image_attention_mask = torch.zeros(batch_size, 1, device=device)
        
        # Process audio modality
        if audio is not None:
            audio_features = self.audio_encoder(audio)
            audio_features = self.audio_position_embeddings(audio_features)
            
            # Add modality token if configured
            if self.config.use_modality_tokens:
                modality_token = self.modality_tokens[2].unsqueeze(0).expand(batch_size, 1, -1)
                audio_features = torch.cat([modality_token, audio_features], dim=1)
                
                # Update attention mask if present
                if audio_attention_mask is not None:
                    modality_token_mask = torch.ones(batch_size, 1, device=device)
                    audio_attention_mask = torch.cat([modality_token_mask, audio_attention_mask], dim=1)
        else:
            # Create empty tensor with proper dimensions for absent modality
            audio_features = torch.zeros(
                batch_size, 
                1,  # Just the modality token
                self.config.audio_hidden_size, 
                device=device
            )
            audio_attention_mask = torch.zeros(batch_size, 1, device=device)
        
        # Process through transformer layers
        for layer in self.layers:
            text_features, image_features, audio_features = layer(
                text_features=text_features,
                image_features=image_features,
                audio_features=audio_features,
                text_attention_mask=text_attention_mask,
                image_attention_mask=image_attention_mask,
                audio_attention_mask=audio_attention_mask,
            )
            
        # Fuse modality representations
        fused_features = self.fusion(
            text_features=text_features,
            image_features=image_features,
            audio_features=audio_features,
            text_attention_mask=text_attention_mask,
            image_attention_mask=image_attention_mask,
            audio_attention_mask=audio_attention_mask,
        )
        
        # Pool sequence representations to get fixed-size vectors
        text_pooled = self.pooling(text_features, text_attention_mask)
        image_pooled = self.pooling(image_features, image_attention_mask)
        audio_pooled = self.pooling(audio_features, audio_attention_mask)
        fused_pooled = self.pooling(fused_features)
        
        if return_dict:
            return {
                "text_features": text_features,
                "image_features": image_features,
                "audio_features": audio_features,
                "fused_features": fused_features,
                "text_pooled": text_pooled,
                "image_pooled": image_pooled,
                "audio_pooled": audio_pooled,
                "fused_pooled": fused_pooled,
            }
        else:
            return fused_pooled
        
    def _get_batch_size(self, text, image, audio):
        """Determine batch size from available inputs"""
        if text is not None:
            return text.size(0)
        elif image is not None:
            return image.size(0)
        elif audio is not None:
            return audio.size(0)
        else:
            raise ValueError("At least one modality input must be provided")
            
    def _get_device(self, text, image, audio):
        """Determine device from available inputs"""
        if text is not None:
            return text.device
        elif image is not None:
            return image.device
        elif audio is not None:
            return audio.device
        else:
            return next(self.parameters()).device
    
    @classmethod
    def from_pretrained(cls, model_name: str) -> "MultiModalTransformer":
        """
        Load a pretrained model from a checkpoint.
        
        Args:
            model_name: Name of the pretrained model
            
        Returns:
            Loaded model instance
        """
        # This would typically load from Hugging Face Hub or similar
        # For now, we'll just return a model with the default config
        config = MultiModalTransformerConfig()
        
        if model_name == "mmtransformer-base":
            pass  # Use default config
        elif model_name == "mmtransformer-large":
            config.hidden_size = 1024
            config.num_hidden_layers = 24
            config.num_attention_heads = 16
            config.intermediate_size = 4096
            config.text_hidden_size = 1024
            config.text_num_hidden_layers = 24
            config.text_num_attention_heads = 16
            config.image_hidden_size = 1024
            config.image_num_hidden_layers = 24
            config.image_num_attention_heads = 16
            config.audio_hidden_size = 1024
            config.audio_num_hidden_layers = 24
            config.audio_num_attention_heads = 16
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        model = cls(config)
        
        # In a real implementation, we would load the weights here
        # model.load_state_dict(torch.load(model_path))
        
        return model 