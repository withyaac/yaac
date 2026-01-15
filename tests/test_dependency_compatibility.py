"""Test that the package works with relaxed dependency versions.

This test verifies that the package works with the minimum and current
versions of dependencies to ensure compatibility across the version ranges.
"""

import torch
import torchvision
import transformers
import numpy
from safetensors.torch import load_file

from yaac.models.sic.sic import make_model
from yaac.common.model_loader import load_model_from_checkpoint


def test_basic_imports():
    """Test that all required packages can be imported."""
    assert torch is not None
    assert torchvision is not None
    assert transformers is not None
    assert numpy is not None


def test_model_creation():
    """Test that models can be created with current dependency versions."""
    # Test ResNet18 backbone
    model = make_model(num_classes=2, backbone_type="resnet18")
    assert model is not None
    
    # Test forward pass
    dummy_image = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_image)
        assert output.shape == (1, 1)  # Binary classification


def test_torchvision_api():
    """Test that torchvision API we use still works."""
    # Test ResNet18 weights API
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    assert weights is not None
    
    # Test model creation
    model = torchvision.models.resnet18(weights=None)
    assert model is not None


def test_safetensors_api():
    """Test that safetensors API works."""
    # Just verify the import works
    assert load_file is not None


def test_transformers_api():
    """Test that transformers AutoModel import works."""
    from transformers import AutoModel
    assert AutoModel is not None
