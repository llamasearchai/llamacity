from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Literal
import json
import os
from copy import deepcopy


@dataclass
class AttentionConfig:
    """
    Configuration for attention mechanisms in the model.
    """
    attention_type: str = "multi_head"  # Options: multi_head, linear, sparse, cross
    num_heads: int = 8
    dropout: float = 0.1
    
    # Parameters for sparse attention
    sparsity_type: Optional[str] = None  # Options: fixed, adaptive, longformer, bigbird
    window_size: Optional[int] = None  # For local windows in sparse attention
    num_global_tokens: Optional[int] = None  # For sparse attention patterns
    
    # Parameters for linear attention
    kernel_type: Optional[str] = None  # Options: elu, relu, softmax
    feature_dim: Optional[int] = None  # Feature dimension for linear attention kernel
    causal: bool = False  # Whether attention is causal (for autoregressive models)
    
    # Parameters for multi-query attention
    multi_query: bool = False  # Whether to use multi-query attention
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "AttentionConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


@dataclass
class EncoderConfig:
    """
    Configuration for transformer encoder.
    """
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    activation: str = "gelu"  # Options: gelu, relu, swish, silu
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    
    # Attention configuration
    attention_config: AttentionConfig = field(default_factory=AttentionConfig)
    
    # Feed-forward configuration
    feed_forward_type: str = "default"  # Options: default, glu, swiglu
    
    # Positional encoding configuration
    positional_encoding_type: str = "learned"  # Options: sinusoidal, learned, rotary, relative, alibi
    max_position_embeddings: int = 512
    
    # Layer organization
    pre_norm: bool = True  # Pre-LayerNorm vs Post-LayerNorm
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization."""
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if k != "attention_config"}
        config_dict["attention_config"] = self.attention_config.to_dict()
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "EncoderConfig":
        """Create config from dictionary."""
        attention_config = config_dict.pop("attention_config", {})
        config = cls(**config_dict)
        config.attention_config = AttentionConfig.from_dict(attention_config)
        return config


@dataclass
class DecoderConfig(EncoderConfig):
    """
    Configuration for transformer decoder.
    Extends EncoderConfig with decoder-specific parameters.
    """
    cross_attention_config: AttentionConfig = field(default_factory=AttentionConfig)
    use_cache: bool = True  # Use key-value cache for faster autoregressive decoding
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization."""
        config_dict = super().to_dict()
        config_dict["cross_attention_config"] = self.cross_attention_config.to_dict()
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "DecoderConfig":
        """Create config from dictionary."""
        cross_attention_config = config_dict.pop("cross_attention_config", {})
        attention_config = config_dict.pop("attention_config", {})
        
        config = cls(**config_dict)
        config.attention_config = AttentionConfig.from_dict(attention_config)
        config.cross_attention_config = AttentionConfig.from_dict(cross_attention_config)
        return config


@dataclass
class FusionConfig:
    """
    Configuration for multi-modal fusion mechanisms.
    """
    fusion_type: str = "cross_attention"  # Options: cross_attention, concat, weighted, gated
    hidden_size: int = 768
    num_attention_heads: int = 8
    dropout: float = 0.1
    
    # For weighted fusion
    learn_weights: bool = True
    
    # For cross-attention fusion
    cross_attention_config: AttentionConfig = field(default_factory=AttentionConfig)
    
    # For gated fusion
    use_gate_activation: str = "sigmoid"  # Options: sigmoid, tanh, relu
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization."""
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if k != "cross_attention_config"}
        config_dict["cross_attention_config"] = self.cross_attention_config.to_dict()
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "FusionConfig":
        """Create config from dictionary."""
        cross_attention_config = config_dict.pop("cross_attention_config", {})
        config = cls(**config_dict)
        config.cross_attention_config = AttentionConfig.from_dict(cross_attention_config)
        return config


@dataclass
class MultiModalConfig:
    """
    Configuration for multi-modal encoders and fusion.
    """
    # Modality-specific configurations
    text_encoder: EncoderConfig = field(default_factory=EncoderConfig)
    image_encoder: Optional[EncoderConfig] = None
    audio_encoder: Optional[EncoderConfig] = None
    video_encoder: Optional[EncoderConfig] = None
    
    # Fusion configuration
    fusion_config: FusionConfig = field(default_factory=FusionConfig)
    
    # Pooling configuration
    pooling_type: str = "attentional"  # Options: attentional, cls, mean, max, hierarchical
    pooling_heads: int = 8
    
    # Whether to use shared parameters across modalities
    use_shared_encoder: bool = False
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization."""
        config_dict = {
            "text_encoder": self.text_encoder.to_dict(),
            "fusion_config": self.fusion_config.to_dict(),
            "pooling_type": self.pooling_type,
            "pooling_heads": self.pooling_heads,
            "use_shared_encoder": self.use_shared_encoder
        }
        
        if self.image_encoder:
            config_dict["image_encoder"] = self.image_encoder.to_dict()
        if self.audio_encoder:
            config_dict["audio_encoder"] = self.audio_encoder.to_dict()
        if self.video_encoder:
            config_dict["video_encoder"] = self.video_encoder.to_dict()
            
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "MultiModalConfig":
        """Create config from dictionary."""
        # Create deepcopy to avoid modifying the input dictionary
        config_copy = deepcopy(config_dict)
        
        # Handle encoder configurations
        text_encoder_dict = config_copy.pop("text_encoder", {})
        text_encoder = EncoderConfig.from_dict(text_encoder_dict)
        
        image_encoder_dict = config_copy.pop("image_encoder", None)
        image_encoder = EncoderConfig.from_dict(image_encoder_dict) if image_encoder_dict else None
        
        audio_encoder_dict = config_copy.pop("audio_encoder", None)
        audio_encoder = EncoderConfig.from_dict(audio_encoder_dict) if audio_encoder_dict else None
        
        video_encoder_dict = config_copy.pop("video_encoder", None)
        video_encoder = EncoderConfig.from_dict(video_encoder_dict) if video_encoder_dict else None
        
        # Handle fusion configuration
        fusion_config_dict = config_copy.pop("fusion_config", {})
        fusion_config = FusionConfig.from_dict(fusion_config_dict)
        
        # Create instance with remaining parameters
        config = cls(
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            audio_encoder=audio_encoder,
            video_encoder=video_encoder,
            fusion_config=fusion_config,
            **config_copy
        )
        
        return config


@dataclass
class ModelConfig:
    """
    Main configuration class for the multi-modal transformer model.
    """
    model_type: str = "encoder_only"  # Options: encoder_only, encoder_decoder, decoder_only
    
    # Encoder configuration
    encoder_config: Optional[EncoderConfig] = field(default_factory=EncoderConfig)
    
    # Decoder configuration (for encoder-decoder models)
    decoder_config: Optional[DecoderConfig] = None
    
    # Multi-modal configuration (for multi-modal models)
    multi_modal_config: Optional[MultiModalConfig] = None
    
    # Tokenizer configurations
    vocab_size: int = 30522
    pad_token_id: int = 0
    bos_token_id: int = 101
    eos_token_id: int = 102
    
    # Model parameters
    hidden_size: int = 768
    initializer_range: float = 0.02
    
    # Task-specific parameters
    task_specific_params: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Run after initialization to ensure consistency."""
        # If encoder_only or encoder_decoder, ensure encoder_config exists
        if self.model_type in ["encoder_only", "encoder_decoder"] and not self.encoder_config:
            self.encoder_config = EncoderConfig(hidden_size=self.hidden_size)
            
        # If encoder_decoder, ensure decoder_config exists
        if self.model_type == "encoder_decoder" and not self.decoder_config:
            self.decoder_config = DecoderConfig(hidden_size=self.hidden_size)
            
        # If decoder_only, ensure decoder_config exists and encoder_config is None
        if self.model_type == "decoder_only":
            if not self.decoder_config:
                self.decoder_config = DecoderConfig(hidden_size=self.hidden_size)
            self.encoder_config = None
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization."""
        config_dict = {
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "hidden_size": self.hidden_size,
            "initializer_range": self.initializer_range,
            "task_specific_params": self.task_specific_params,
        }
        
        if self.encoder_config:
            config_dict["encoder_config"] = self.encoder_config.to_dict()
        if self.decoder_config:
            config_dict["decoder_config"] = self.decoder_config.to_dict()
        if self.multi_modal_config:
            config_dict["multi_modal_config"] = self.multi_modal_config.to_dict()
            
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "ModelConfig":
        """Create config from dictionary."""
        # Create deepcopy to avoid modifying the input dictionary
        config_copy = deepcopy(config_dict)
        
        # Handle nested configurations
        encoder_config_dict = config_copy.pop("encoder_config", None)
        encoder_config = EncoderConfig.from_dict(encoder_config_dict) if encoder_config_dict else None
        
        decoder_config_dict = config_copy.pop("decoder_config", None)
        decoder_config = DecoderConfig.from_dict(decoder_config_dict) if decoder_config_dict else None
        
        multi_modal_config_dict = config_copy.pop("multi_modal_config", None)
        multi_modal_config = MultiModalConfig.from_dict(multi_modal_config_dict) if multi_modal_config_dict else None
        
        # Create instance with remaining parameters
        config = cls(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            multi_modal_config=multi_modal_config,
            **config_copy
        )
        
        return config
    
    @classmethod
    def from_json_file(cls, json_file: str) -> "ModelConfig":
        """Load configuration from JSON file."""
        with open(json_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def save_pretrained(self, save_directory: str):
        """Save configuration to JSON file in the specified directory."""
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            
        config_file = os.path.join(save_directory, "config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def from_pretrained(cls, model_path: str) -> "ModelConfig":
        """Load configuration from pretrained model directory."""
        config_file = os.path.join(model_path, "config.json")
        return cls.from_json_file(config_file)
    
    def create_default_multi_modal_config(self, 
                                          modalities: List[str] = ["text", "image"],
                                          hidden_size: Optional[int] = None):
        """Create a default multi-modal configuration."""
        if hidden_size is None:
            hidden_size = self.hidden_size
            
        # Create base encoder config
        base_encoder = EncoderConfig(hidden_size=hidden_size)
        
        # Create multi-modal config
        multi_modal_config = MultiModalConfig(
            text_encoder=deepcopy(base_encoder)
        )
        
        # Add other modality encoders
        if "image" in modalities:
            multi_modal_config.image_encoder = deepcopy(base_encoder)
        if "audio" in modalities:
            multi_modal_config.audio_encoder = deepcopy(base_encoder)
        if "video" in modalities:
            multi_modal_config.video_encoder = deepcopy(base_encoder)
            
        # Set fusion config
        multi_modal_config.fusion_config = FusionConfig(hidden_size=hidden_size)
        
        self.multi_modal_config = multi_modal_config
        return self 