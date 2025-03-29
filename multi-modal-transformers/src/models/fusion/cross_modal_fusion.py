import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math
import einops

from ..layers import LayerNorm, FeedForward
from ..attention import MultiHeadCrossAttention


class CrossModalFusion(nn.Module):
    """
    Cross-Modal Fusion module that combines representations from multiple modalities.
    Implements various fusion strategies, including:
    - Cross-attention fusion
    - Concatenation with projection
    - Weighted sum with learnable weights
    - Gated fusion
    """
    
    def __init__(self, config):
        """
        Initialize cross-modal fusion module.
        
        Args:
            config: Configuration object containing fusion parameters
        """
        super().__init__()
        self.config = config
        self.fusion_type = config.fusion_type
        self.hidden_size = config.hidden_size
        
        # Common components
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Components for cross-attention fusion
        if self.fusion_type == "cross_attention":
            self.fusion_layers = nn.ModuleList([
                CrossAttentionFusionLayer(config)
                for _ in range(config.fusion_layers)
            ])
        
        # Components for concatenation fusion
        elif self.fusion_type == "concat":
            # Calculate total size from all modalities
            total_size = config.text_hidden_size + config.image_hidden_size + config.audio_hidden_size
            
            # Create projection layer
            self.fusion_projection = nn.Sequential(
                nn.Linear(total_size, config.hidden_size * 2),
                nn.GELU(),
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size * 2, config.hidden_size)
            )
            
            # Layer norm for final output
            self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Components for weighted sum fusion
        elif self.fusion_type == "weighted":
            # Create learnable weights for each modality
            self.fusion_weights = nn.Parameter(torch.ones(3))
            
            # Projection layers to common dimension if necessary
            self.text_projection = nn.Linear(config.text_hidden_size, config.hidden_size) \
                if config.text_hidden_size != config.hidden_size else nn.Identity()
            
            self.image_projection = nn.Linear(config.image_hidden_size, config.hidden_size) \
                if config.image_hidden_size != config.hidden_size else nn.Identity()
            
            self.audio_projection = nn.Linear(config.audio_hidden_size, config.hidden_size) \
                if config.audio_hidden_size != config.hidden_size else nn.Identity()
            
            # Layer norm for final output
            self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Components for gated fusion
        elif self.fusion_type == "gated":
            # Project each modality to common dimension if necessary
            self.text_projection = nn.Linear(config.text_hidden_size, config.hidden_size) \
                if config.text_hidden_size != config.hidden_size else nn.Identity()
            
            self.image_projection = nn.Linear(config.image_hidden_size, config.hidden_size) \
                if config.image_hidden_size != config.hidden_size else nn.Identity()
            
            self.audio_projection = nn.Linear(config.audio_hidden_size, config.hidden_size) \
                if config.audio_hidden_size != config.hidden_size else nn.Identity()
            
            # Gate networks for each modality
            self.text_gate = nn.Sequential(
                nn.Linear(config.hidden_size * 3, config.hidden_size),
                nn.Sigmoid()
            )
            
            self.image_gate = nn.Sequential(
                nn.Linear(config.hidden_size * 3, config.hidden_size),
                nn.Sigmoid()
            )
            
            self.audio_gate = nn.Sequential(
                nn.Linear(config.hidden_size * 3, config.hidden_size),
                nn.Sigmoid()
            )
            
            # Final projection
            self.fusion_projection = nn.Linear(config.hidden_size, config.hidden_size)
            
            # Layer norm for final output
            self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
    
    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        audio_features: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply cross-modal fusion to input features.
        
        Args:
            text_features: Text embeddings [batch_size, text_seq_len, text_hidden_size]
            image_features: Image embeddings [batch_size, image_seq_len, image_hidden_size]
            audio_features: Audio embeddings [batch_size, audio_seq_len, audio_hidden_size]
            text_attention_mask: Mask for text inputs [batch_size, text_seq_len]
            image_attention_mask: Mask for image inputs [batch_size, image_seq_len]
            audio_attention_mask: Mask for audio inputs [batch_size, audio_seq_len]
            
        Returns:
            Fused representation [batch_size, fusion_seq_len, hidden_size]
        """
        # Cross-attention fusion
        if self.fusion_type == "cross_attention":
            fused_features = self._cross_attention_fusion(
                text_features, image_features, audio_features,
                text_attention_mask, image_attention_mask, audio_attention_mask
            )
        
        # Concatenation fusion
        elif self.fusion_type == "concat":
            fused_features = self._concat_fusion(
                text_features, image_features, audio_features,
                text_attention_mask, image_attention_mask, audio_attention_mask
            )
        
        # Weighted sum fusion
        elif self.fusion_type == "weighted":
            fused_features = self._weighted_fusion(
                text_features, image_features, audio_features,
                text_attention_mask, image_attention_mask, audio_attention_mask
            )
        
        # Gated fusion
        elif self.fusion_type == "gated":
            fused_features = self._gated_fusion(
                text_features, image_features, audio_features,
                text_attention_mask, image_attention_mask, audio_attention_mask
            )
        
        return fused_features
    
    def _cross_attention_fusion(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        audio_features: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply cross-attention fusion to combine modalities.
        
        Applies multiple layers of cross-attention between all modalities.
        """
        # Concatenate all modality features as the initial fusion representation
        # This becomes the query for cross-attention
        batch_size = text_features.shape[0]
        device = text_features.device
        
        # Create fusion representation with modality tokens
        if self.config.use_modality_tokens:
            # Use the first token of each modality sequence
            fusion_features = torch.cat([
                text_features[:, 0:1],
                image_features[:, 0:1],
                audio_features[:, 0:1]
            ], dim=1)
            
            # Create corresponding attention mask
            fusion_attention_mask = torch.ones(batch_size, 3, device=device)
            
            # Process through fusion layers
            for layer in self.fusion_layers:
                fusion_features = layer(
                    fusion_features=fusion_features,
                    text_features=text_features,
                    image_features=image_features,
                    audio_features=audio_features,
                    fusion_attention_mask=fusion_attention_mask,
                    text_attention_mask=text_attention_mask,
                    image_attention_mask=image_attention_mask,
                    audio_attention_mask=audio_attention_mask,
                )
        else:
            # Concatenate all sequences
            fusion_features = torch.cat([text_features, image_features, audio_features], dim=1)
            
            # Concatenate attention masks
            if text_attention_mask is not None and image_attention_mask is not None and audio_attention_mask is not None:
                fusion_attention_mask = torch.cat([
                    text_attention_mask, image_attention_mask, audio_attention_mask
                ], dim=1)
            else:
                text_seq_len = text_features.shape[1]
                image_seq_len = image_features.shape[1]
                audio_seq_len = audio_features.shape[1]
                fusion_attention_mask = torch.ones(
                    batch_size, text_seq_len + image_seq_len + audio_seq_len, device=device
                )
            
            # Process through fusion layers
            for layer in self.fusion_layers:
                fusion_features = layer(
                    fusion_features=fusion_features,
                    text_features=text_features,
                    image_features=image_features,
                    audio_features=audio_features,
                    fusion_attention_mask=fusion_attention_mask,
                    text_attention_mask=text_attention_mask,
                    image_attention_mask=image_attention_mask,
                    audio_attention_mask=audio_attention_mask,
                )
        
        return fusion_features
    
    def _concat_fusion(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        audio_features: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply concatenation fusion to combine modalities.
        
        Concatenates all modality features and projects to desired dimension.
        """
        batch_size = text_features.shape[0]
        device = text_features.device
        
        # Use first token (CLS) from each modality if modality tokens are enabled
        if self.config.use_modality_tokens:
            # Extract first token from each modality
            text_cls = text_features[:, 0]
            image_cls = image_features[:, 0]
            audio_cls = audio_features[:, 0]
            
            # Concatenate along feature dimension
            concat_features = torch.cat([text_cls, image_cls, audio_cls], dim=-1)
            
            # Project to target dimension
            fused_features = self.fusion_projection(concat_features)
            fused_features = self.layer_norm(fused_features)
            
            # Add batch dimension for consistency
            fused_features = fused_features.unsqueeze(1)
        else:
            # For sequence-based fusion, align sequence lengths
            # Use mean pooling for each modality based on attention masks
            if text_attention_mask is not None:
                text_mask = text_attention_mask.unsqueeze(-1)
                text_sum = (text_features * text_mask).sum(dim=1)
                text_count = text_mask.sum(dim=1)
                text_pooled = text_sum / (text_count + 1e-6)
            else:
                text_pooled = text_features.mean(dim=1)
            
            if image_attention_mask is not None:
                image_mask = image_attention_mask.unsqueeze(-1)
                image_sum = (image_features * image_mask).sum(dim=1)
                image_count = image_mask.sum(dim=1)
                image_pooled = image_sum / (image_count + 1e-6)
            else:
                image_pooled = image_features.mean(dim=1)
            
            if audio_attention_mask is not None:
                audio_mask = audio_attention_mask.unsqueeze(-1)
                audio_sum = (audio_features * audio_mask).sum(dim=1)
                audio_count = audio_mask.sum(dim=1)
                audio_pooled = audio_sum / (audio_count + 1e-6)
            else:
                audio_pooled = audio_features.mean(dim=1)
            
            # Concatenate along feature dimension
            concat_features = torch.cat([text_pooled, image_pooled, audio_pooled], dim=-1)
            
            # Project to target dimension
            fused_features = self.fusion_projection(concat_features)
            fused_features = self.layer_norm(fused_features)
            
            # Add sequence dimension for consistency
            fused_features = fused_features.unsqueeze(1)
        
        return fused_features
    
    def _weighted_fusion(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        audio_features: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply weighted sum fusion to combine modalities.
        
        Uses learnable weights to combine the modalities.
        """
        batch_size = text_features.shape[0]
        device = text_features.device
        
        # Project each modality to common dimension
        text_proj = self.text_projection(text_features)
        image_proj = self.image_projection(image_features)
        audio_proj = self.audio_projection(audio_features)
        
        # Apply softmax to get normalized weights
        fusion_weights = F.softmax(self.fusion_weights, dim=0)
        
        # Use first token (CLS) from each modality if modality tokens are enabled
        if self.config.use_modality_tokens:
            # Extract first token from each modality
            text_cls = text_proj[:, 0]
            image_cls = image_proj[:, 0]
            audio_cls = audio_proj[:, 0]
            
            # Apply weighted sum
            fused_features = (
                fusion_weights[0] * text_cls +
                fusion_weights[1] * image_cls +
                fusion_weights[2] * audio_cls
            )
            
            # Apply layer norm
            fused_features = self.layer_norm(fused_features)
            
            # Add sequence dimension for consistency
            fused_features = fused_features.unsqueeze(1)
        else:
            # For sequence-based fusion, align sequence lengths
            # Use mean pooling for each modality based on attention masks
            if text_attention_mask is not None:
                text_mask = text_attention_mask.unsqueeze(-1)
                text_sum = (text_proj * text_mask).sum(dim=1)
                text_count = text_mask.sum(dim=1)
                text_pooled = text_sum / (text_count + 1e-6)
            else:
                text_pooled = text_proj.mean(dim=1)
            
            if image_attention_mask is not None:
                image_mask = image_attention_mask.unsqueeze(-1)
                image_sum = (image_proj * image_mask).sum(dim=1)
                image_count = image_mask.sum(dim=1)
                image_pooled = image_sum / (image_count + 1e-6)
            else:
                image_pooled = image_proj.mean(dim=1)
            
            if audio_attention_mask is not None:
                audio_mask = audio_attention_mask.unsqueeze(-1)
                audio_sum = (audio_proj * audio_mask).sum(dim=1)
                audio_count = audio_mask.sum(dim=1)
                audio_pooled = audio_sum / (audio_count + 1e-6)
            else:
                audio_pooled = audio_proj.mean(dim=1)
            
            # Apply weighted sum
            fused_features = (
                fusion_weights[0] * text_pooled +
                fusion_weights[1] * image_pooled +
                fusion_weights[2] * audio_pooled
            )
            
            # Apply layer norm
            fused_features = self.layer_norm(fused_features)
            
            # Add sequence dimension for consistency
            fused_features = fused_features.unsqueeze(1)
        
        return fused_features
    
    def _gated_fusion(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        audio_features: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply gated fusion to combine modalities.
        
        Uses dynamic gates to control the contribution of each modality.
        """
        batch_size = text_features.shape[0]
        device = text_features.device
        
        # Project each modality to common dimension
        text_proj = self.text_projection(text_features)
        image_proj = self.image_projection(image_features)
        audio_proj = self.audio_projection(audio_features)
        
        # Use first token (CLS) from each modality if modality tokens are enabled
        if self.config.use_modality_tokens:
            # Extract first token from each modality
            text_cls = text_proj[:, 0]
            image_cls = image_proj[:, 0]
            audio_cls = audio_proj[:, 0]
            
            # Concatenate features for gate computation
            concat_features = torch.cat([text_cls, image_cls, audio_cls], dim=-1)
            
            # Compute gates for each modality
            text_gate_value = self.text_gate(concat_features)
            image_gate_value = self.image_gate(concat_features)
            audio_gate_value = self.audio_gate(concat_features)
            
            # Apply gated fusion
            fused_features = (
                text_gate_value * text_cls +
                image_gate_value * image_cls +
                audio_gate_value * audio_cls
            )
            
            # Final projection and normalization
            fused_features = self.fusion_projection(fused_features)
            fused_features = self.layer_norm(fused_features)
            
            # Add sequence dimension for consistency
            fused_features = fused_features.unsqueeze(1)
        else:
            # For sequence-based fusion, align sequence lengths
            # Use mean pooling for each modality based on attention masks
            if text_attention_mask is not None:
                text_mask = text_attention_mask.unsqueeze(-1)
                text_sum = (text_proj * text_mask).sum(dim=1)
                text_count = text_mask.sum(dim=1)
                text_pooled = text_sum / (text_count + 1e-6)
            else:
                text_pooled = text_proj.mean(dim=1)
            
            if image_attention_mask is not None:
                image_mask = image_attention_mask.unsqueeze(-1)
                image_sum = (image_proj * image_mask).sum(dim=1)
                image_count = image_mask.sum(dim=1)
                image_pooled = image_sum / (image_count + 1e-6)
            else:
                image_pooled = image_proj.mean(dim=1)
            
            if audio_attention_mask is not None:
                audio_mask = audio_attention_mask.unsqueeze(-1)
                audio_sum = (audio_proj * audio_mask).sum(dim=1)
                audio_count = audio_mask.sum(dim=1)
                audio_pooled = audio_sum / (audio_count + 1e-6)
            else:
                audio_pooled = audio_proj.mean(dim=1)
            
            # Concatenate features for gate computation
            concat_features = torch.cat([text_pooled, image_pooled, audio_pooled], dim=-1)
            
            # Compute gates for each modality
            text_gate_value = self.text_gate(concat_features)
            image_gate_value = self.image_gate(concat_features)
            audio_gate_value = self.audio_gate(concat_features)
            
            # Apply gated fusion
            fused_features = (
                text_gate_value * text_pooled +
                image_gate_value * image_pooled +
                audio_gate_value * audio_pooled
            )
            
            # Final projection and normalization
            fused_features = self.fusion_projection(fused_features)
            fused_features = self.layer_norm(fused_features)
            
            # Add sequence dimension for consistency
            fused_features = fused_features.unsqueeze(1)
        
        return fused_features


class CrossAttentionFusionLayer(nn.Module):
    """
    Cross-attention fusion layer that enables interaction between fused representation
    and individual modality encodings.
    """
    
    def __init__(self, config):
        """
        Initialize cross-attention fusion layer.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config
        
        # Self-attention for fusion representation
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True
        )
        
        # Cross-attention mechanisms to each modality
        self.text_cross_attention = MultiHeadCrossAttention(
            query_dim=config.hidden_size,
            key_dim=config.text_hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob
        )
        
        self.image_cross_attention = MultiHeadCrossAttention(
            query_dim=config.hidden_size,
            key_dim=config.image_hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob
        )
        
        self.audio_cross_attention = MultiHeadCrossAttention(
            query_dim=config.hidden_size,
            key_dim=config.audio_hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob
        )
        
        # Feed-forward network
        self.feed_forward = FeedForward(
            d_model=config.hidden_size,
            d_ff=config.intermediate_size,
            dropout=config.hidden_dropout_prob
        )
        
        # Layer normalization
        self.layer_norm1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm3 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(
        self,
        fusion_features: torch.Tensor,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        audio_features: torch.Tensor,
        fusion_attention_mask: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process the fusion representation through self-attention and cross-attention.
        
        Args:
            fusion_features: Fused representation [batch_size, fusion_seq_len, hidden_size]
            text_features: Text features [batch_size, text_seq_len, text_hidden_size]
            image_features: Image features [batch_size, image_seq_len, image_hidden_size]
            audio_features: Audio features [batch_size, audio_seq_len, audio_hidden_size]
            fusion_attention_mask: Attention mask for fusion sequence [batch_size, fusion_seq_len]
            text_attention_mask: Attention mask for text [batch_size, text_seq_len]
            image_attention_mask: Attention mask for image [batch_size, image_seq_len]
            audio_attention_mask: Attention mask for audio [batch_size, audio_seq_len]
            
        Returns:
            Updated fusion representation [batch_size, fusion_seq_len, hidden_size]
        """
        # Apply self-attention to fusion representation
        residual = fusion_features
        fusion_features = self.layer_norm1(fusion_features)
        
        # Prepare attention mask for self-attention
        if fusion_attention_mask is not None:
            attention_mask = ~fusion_attention_mask.bool()
        else:
            attention_mask = None
        
        fusion_features, _ = self.self_attention(
            query=fusion_features,
            key=fusion_features,
            value=fusion_features,
            key_padding_mask=attention_mask
        )
        fusion_features = self.dropout(fusion_features)
        fusion_features = residual + fusion_features
        
        # Apply cross-attention to each modality
        residual = fusion_features
        fusion_features = self.layer_norm2(fusion_features)
        
        # Cross-attend to text
        text_context = self.text_cross_attention(
            query=fusion_features,
            key=text_features,
            value=text_features,
            key_mask=text_attention_mask
        )
        
        # Cross-attend to image
        image_context = self.image_cross_attention(
            query=fusion_features,
            key=image_features,
            value=image_features,
            key_mask=image_attention_mask
        )
        
        # Cross-attend to audio
        audio_context = self.audio_cross_attention(
            query=fusion_features,
            key=audio_features,
            value=audio_features,
            key_mask=audio_attention_mask
        )
        
        # Combine cross-attention outputs
        cross_attention_output = text_context + image_context + audio_context
        cross_attention_output = self.dropout(cross_attention_output)
        fusion_features = residual + cross_attention_output
        
        # Apply feed-forward network
        residual = fusion_features
        fusion_features = self.layer_norm3(fusion_features)
        fusion_features = self.feed_forward(fusion_features)
        fusion_features = self.dropout(fusion_features)
        fusion_features = residual + fusion_features
        
        return fusion_features 