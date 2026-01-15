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
