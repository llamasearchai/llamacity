import os
import json
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any, Type

from src.config import ModelConfig, MultiModalConfig, EncoderConfig, DecoderConfig
from src.models.transformer import (
    TransformerEncoder, 
    TransformerDecoder, 
    MultiModalTransformer
)


class ModelFactory:
    """
    Factory class to create model instances from configurations.
    """
    
    @staticmethod
    def create_encoder(config: EncoderConfig) -> TransformerEncoder:
        """
        Create a transformer encoder from a configuration.
        
        Args:
            config: Encoder configuration
            
        Returns:
            TransformerEncoder instance
        """
        return TransformerEncoder(config)
    
    @staticmethod
    def create_decoder(config: DecoderConfig) -> TransformerDecoder:
        """
        Create a transformer decoder from a configuration.
        
        Args:
            config: Decoder configuration
            
        Returns:
            TransformerDecoder instance
        """
        return TransformerDecoder(config)
    
    @staticmethod
    def create_multi_modal_transformer(config: MultiModalConfig) -> MultiModalTransformer:
        """
        Create a multi-modal transformer from a configuration.
        
        Args:
            config: Multi-modal configuration
            
        Returns:
            MultiModalTransformer instance
        """
        return MultiModalTransformer(config)
    
    @classmethod
    def from_config(cls, config: ModelConfig) -> nn.Module:
        """
        Create a model from a configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            Model instance based on the configuration
        """
        if config.model_type == "encoder_only":
            if config.multi_modal_config is not None:
                return cls.create_multi_modal_transformer(config.multi_modal_config)
            else:
                assert config.encoder_config is not None, "Encoder config must be provided"
                return cls.create_encoder(config.encoder_config)
        
        elif config.model_type == "decoder_only":
            assert config.decoder_config is not None, "Decoder config must be provided"
            return cls.create_decoder(config.decoder_config)
        
        elif config.model_type == "encoder_decoder":
            raise NotImplementedError("Encoder-decoder models not yet implemented")
        
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
    
    @classmethod
    def from_pretrained(cls, model_path: str) -> nn.Module:
        """
        Load a model from a pretrained checkpoint.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            Loaded model instance
        """
        # Load configuration
        config = ModelConfig.from_pretrained(model_path)
        
        # Create model from configuration
        model = cls.from_config(config)
        
        # Load weights
        checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict)
        
        return model
    
    @classmethod
    def from_json_file(cls, json_file: str) -> nn.Module:
        """
        Create a model from a JSON configuration file.
        
        Args:
            json_file: Path to the JSON configuration file
            
        Returns:
            Model instance based on the configuration
        """
        config = ModelConfig.from_json_file(json_file)
        return cls.from_config(config)


class ModelRegistry:
    """
    Registry for model architectures to enable easy instantiation.
    """
    _registry: Dict[str, Type[nn.Module]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a model class."""
        def wrapper(model_cls):
            cls._registry[name] = model_cls
            return model_cls
        return wrapper
    
    @classmethod
    def get_model_class(cls, name: str) -> Type[nn.Module]:
        """Get a model class by name."""
        if name not in cls._registry:
            raise ValueError(f"Model {name} not found in registry")
        return cls._registry[name]
    
    @classmethod
    def create(cls, name: str, *args, **kwargs) -> nn.Module:
        """Create a model instance by name."""
        model_cls = cls.get_model_class(name)
        return model_cls(*args, **kwargs)


# Register standard model types
@ModelRegistry.register("transformer_encoder")
class TransformerEncoderWrapper(nn.Module):
    def __init__(self, config: Union[Dict, EncoderConfig]):
        super().__init__()
        if isinstance(config, dict):
            config = EncoderConfig.from_dict(config)
        self.encoder = ModelFactory.create_encoder(config)
    
    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)


@ModelRegistry.register("transformer_decoder")
class TransformerDecoderWrapper(nn.Module):
    def __init__(self, config: Union[Dict, DecoderConfig]):
        super().__init__()
        if isinstance(config, dict):
            config = DecoderConfig.from_dict(config)
        self.decoder = ModelFactory.create_decoder(config)
    
    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


@ModelRegistry.register("multi_modal_transformer")
class MultiModalTransformerWrapper(nn.Module):
    def __init__(self, config: Union[Dict, MultiModalConfig]):
        super().__init__()
        if isinstance(config, dict):
            config = MultiModalConfig.from_dict(config)
        self.model = ModelFactory.create_multi_modal_transformer(config)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


# Helper function to simplify model creation
def create_model(
    model_name_or_path: str,
    model_type: Optional[str] = None,
    **kwargs
) -> nn.Module:
    """
    Create a model from a name, path, or configuration.
    
    Args:
        model_name_or_path: Model name, path, or configuration
        model_type: Model type (if creating from registry)
        **kwargs: Additional arguments for model creation
        
    Returns:
        Model instance
    """
    # Check if it's a path to a directory with a saved model
    if os.path.isdir(model_name_or_path) and os.path.exists(os.path.join(model_name_or_path, "config.json")):
        return ModelFactory.from_pretrained(model_name_or_path)
    
    # Check if it's a path to a JSON configuration file
    if model_name_or_path.endswith(".json") and os.path.exists(model_name_or_path):
        return ModelFactory.from_json_file(model_name_or_path)
    
    # Check if it's a registered model type
    if model_type is not None:
        return ModelRegistry.create(model_type, model_name_or_path, **kwargs)
    
    # Try to guess the model type
    for model_type in ModelRegistry._registry:
        if model_type in model_name_or_path:
            return ModelRegistry.create(model_type, model_name_or_path, **kwargs)
    
    raise ValueError(
        f"Could not create model from {model_name_or_path}. "
        "Please provide a valid model name, path, or configuration."
    ) 