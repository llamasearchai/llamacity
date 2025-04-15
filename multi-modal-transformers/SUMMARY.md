# Multi-Modal Transformers Repository

This repository contains a comprehensive implementation of a multi-modal transformer architecture with custom attention mechanisms, fusion modules, and training utilities. The repository is designed to be modular, extensible, and easy to use for various multi-modal tasks such as image-text retrieval, visual question answering, and image captioning.

## Repository Structure

```
multi-modal-transformers/
├── configs/                  # Configuration files
│   └── base_text_image_model.json
├── data/                     # Data files and utilities
│   └── sample/               # Sample data for testing
│       ├── images/           # Sample images
│       ├── train.json        # Sample training data
│       └── val.json          # Sample validation data
├── examples/                 # Example scripts
│   ├── create_model.py       # Example script for creating a model
│   └── train_model.py        # Example script for training a model
├── scripts/                  # Utility scripts
│   └── generate_dummy_images.py  # Script to generate dummy images
├── src/                      # Source code
│   ├── config/               # Configuration classes
│   │   ├── __init__.py
│   │   └── model_config.py
│   ├── data/                 # Data loading and processing
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── losses/               # Loss functions
│   │   ├── __init__.py
│   │   ├── contrastive_loss.py
│   │   └── cross_entropy_loss.py
│   ├── models/               # Model implementations
│   │   ├── attention/        # Attention mechanisms
│   │   │   ├── __init__.py
│   │   │   ├── cross_attention.py
│   │   │   ├── linear_attention.py
│   │   │   └── sparse_attention.py
│   │   ├── fusion/           # Fusion modules
│   │   │   ├── __init__.py
│   │   │   ├── attentional_pooling.py
│   │   │   └── cross_modal_fusion.py
│   │   ├── layers/           # Model layers
│   │   │   ├── __init__.py
│   │   │   ├── feed_forward.py
│   │   │   ├── layer_norm.py
│   │   │   └── positional_encoding.py
│   │   └── transformer.py    # Transformer model
│   ├── utils/                # Utility functions
│   │   ├── checkpointing.py  # Checkpointing utilities
│   │   └── logging.py        # Logging utilities
│   └── model_factory.py      # Factory for creating models
├── tests/                    # Tests
├── README.md                 # Repository README
├── requirements.txt          # Package dependencies
└── setup.py                  # Package installation
```

## Key Components

### Attention Mechanisms

The repository includes several advanced attention mechanisms:

1. **Multi-Head Attention**: Standard transformer attention mechanism.
2. **Cross-Attention**: Attention mechanism for cross-modal interactions.
3. **Sparse Attention**: Efficient attention for long sequences with local and global patterns.
4. **Linear Attention**: O(N) complexity attention using kernel methods.

### Fusion Modules

For combining information from different modalities:

1. **Cross-Modal Fusion**: Combines outputs from different modalities using various strategies.
2. **Attentional Pooling**: Converts sequence representations to fixed-size vectors.

### Model Layers

Building blocks for the transformer architecture:

1. **Positional Encoding**: Various positional encoding methods (learned, sinusoidal, rotary, relative).
2. **Feed-Forward Networks**: Standard, GLU, and SwiGLU feed-forward networks.
3. **Layer Normalization**: Various normalization methods (LayerNorm, RMSNorm, ScaleNorm).

### Configuration System

A comprehensive configuration system using dataclasses:

1. **ModelConfig**: Main configuration class for the model.
2. **MultiModalConfig**: Configuration for multi-modal encoders and fusion.
3. **EncoderConfig**: Configuration for transformer encoder.
4. **DecoderConfig**: Configuration for transformer decoder.
5. **AttentionConfig**: Configuration for attention mechanisms.
6. **FusionConfig**: Configuration for fusion mechanisms.

### Training Utilities

Utilities for training and evaluation:

1. **Loss Functions**: Contrastive loss for retrieval, cross-entropy for classification and generation.
2. **Checkpointing**: Utilities for saving and loading model checkpoints.
3. **Logging**: Utilities for logging metrics during training.

### Data Handling

Classes for handling multi-modal data:

1. **MultiModalDataset**: Base class for multi-modal datasets.
2. **ImageTextDataset**: Dataset for image-text retrieval tasks.
3. **VQADataset**: Dataset for visual question answering tasks.
4. **CaptioningDataset**: Dataset for image captioning tasks.

## Usage Examples

### Creating a Model

```python
from src.model_factory import create_model

# Create a model from a configuration file
model = create_model("configs/base_text_image_model.json")

# Use the model for inference
outputs = model(
    text_inputs=text_inputs,
    text_attention_mask=text_attention_mask,
    image_features=image_features,
    image_attention_mask=image_attention_mask
)
```

### Training a Model

```python
python examples/train_model.py \
    --config configs/base_text_image_model.json \
    --task retrieval \
    --train_data data/sample/train.json \
    --val_data data/sample/val.json \
    --output_dir outputs \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 5e-5
```

## Future Work

1. **Encoder-Decoder Models**: Implement encoder-decoder models for generation tasks.
2. **More Attention Mechanisms**: Add more attention mechanisms such as Performer, Reformer, etc.
3. **More Fusion Methods**: Add more fusion methods such as bottleneck fusion, tensor fusion, etc.
4. **More Tasks**: Add support for more tasks such as image-text generation, multi-modal classification, etc.
5. **Distributed Training**: Add support for distributed training.
6. **Quantization**: Add support for model quantization.
7. **Deployment**: Add utilities for model deployment.

## References

1. Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.
2. Dosovitskiy, A., et al. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. ICLR.
3. Radford, A., et al. (2021). Learning transferable visual models from natural language supervision. ICML.
4. Jaegle, A., et al. (2021). Perceiver: General perception with iterative attention. ICML.
5. Katharopoulos, A., et al. (2020). Transformers are RNNs: Fast autoregressive transformers with linear attention. ICML.
6. Child, R., et al. (2019). Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509. 