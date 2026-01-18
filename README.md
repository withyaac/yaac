# yaac

[![PyPI version](https://img.shields.io/pypi/v/yaac.svg)](https://pypi.org/project/yaac/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/yaac.svg)](https://pypi.org/project/yaac/)
[![License](https://img.shields.io/pypi/l/yaac.svg)](https://pypi.org/project/yaac/)

YAAC - Python package for loading and train AI models.

## Installation

Install from PyPI:

```bash
pip install yaac
```

## Requirements

### HuggingFace Token (for DINOv3 ConvNeXt-Tiny backbone)

If you're loading models that use the DINOv3 ConvNeXt-Tiny backbone (`convnext_tiny_dinov3`), you'll need a HuggingFace token because the model repository is gated.

1. **Request access**: Visit https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m and request access to the repository
2. **Generate a token**: Create a token at https://huggingface.co/settings/tokens
3. **Set environment variable**: Export the token as an environment variable:
   ```bash
   export HUGGINGFACE_TOKEN=your_token_here
   ```

The token is only required when loading models with the ConvNeXt-Tiny backbone. Models using ResNet18 or other backbones don't require a token.

## Quick Start

Load a trained model and run inference:

```python
from yaac.common.model_loader import load_model_from_checkpoint
import torch

# Load your trained model
model, config = load_model_from_checkpoint("path/to/checkpoint", device="cuda")

# Run inference
image = torch.randn(1, 3, 224, 224)  # Your image tensor
with torch.no_grad():
    predictions = model(image)
    processed = model.postprocess(predictions)

print(f"Predictions: {processed}")
```

## What is yaac?

`yaac` is a Python package that provides:

- **Model Loading**: Load trained image classification models from exported checkpoints (safetensors + config.json)
- **Model Interface**: Standardized `TrainableModel` interface for consistent model usage
- **SIC Models**: Support for Simple Image Classifier (SIC) models with configurable backbones and heads

Models are trained using YAAC's infrastructure and exported in a format compatible with this package.

## Documentation

- [Contributing Guide](CONTRIBUTING.md) - For developers who want to contribute
- [Publishing Guide](publish/PUBLISHING.md) - For maintainers publishing new versions

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
