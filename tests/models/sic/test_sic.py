"""Tests for the SIC model."""

import pytest
import torch

from yaac.models.sic.sic import SIC, make_model


def test_make_model_shapetype():
    """Test that make_model creates a model with correct type and structure."""
    model = make_model(num_classes=10, pretrained=False)
    assert isinstance(model, SIC)
    assert isinstance(model, torch.nn.Module)


def test_make_model_binary_classification():
    """Test that make_model configures binary classification correctly."""
    model = make_model(num_classes=2, pretrained=False)
    
    # Check that it uses BCEWithLogitsLoss for binary classification
    assert isinstance(model._loss_function, torch.nn.BCEWithLogitsLoss)
    assert model._postprocess_function == torch.sigmoid


def test_make_model_multi_classification():
    """Test that make_model configures multi-class classification correctly."""
    model = make_model(num_classes=5, pretrained=False)
    
    # Check that it uses CrossEntropyLoss for multi-class classification
    assert isinstance(model._loss_function, torch.nn.CrossEntropyLoss)
    assert model._postprocess_function.func == torch.nn.functional.softmax


def test_sic_forward_shapetype_binary():
    """Test that the forward pass outputs have correct shapes and types for binary classification."""
    model = make_model(num_classes=2, pretrained=False)
    batch_size = 3
    image = torch.randn(batch_size, 3, 224, 224)
    outputs = model(image)

    # Check types
    assert isinstance(outputs, torch.Tensor)

    # Check shapes
    assert outputs.shape[0] == batch_size
    assert outputs.shape[1] == 2  # Binary classification outputs 2 logits


def test_sic_forward_shapetype_multi():
    """Test that the forward pass outputs have correct shapes and types for multi-class classification."""
    model = make_model(num_classes=5, pretrained=False)
    batch_size = 4
    image = torch.randn(batch_size, 3, 224, 224)
    outputs = model(image)

    # Check types
    assert isinstance(outputs, torch.Tensor)

    # Check shapes
    assert outputs.shape[0] == batch_size
    assert outputs.shape[1] == 5  # Multi-class classification outputs 5 logits


def test_sic_loss_shapetype_binary():
    """Test that the loss function outputs have correct shapes and types for binary classification."""
    model = make_model(num_classes=2, pretrained=False)
    batch_size = 3
    image = torch.randn(batch_size, 3, 224, 224)
    outputs = model(image)
    
    # Create ground truth targets for binary classification
    targets = torch.randint(0, 2, (batch_size, 2)).float()  # Binary targets
    
    # Call loss function
    losses = model.loss(outputs, targets)

    # Check types
    assert isinstance(losses, dict)
    assert all(isinstance(v, torch.Tensor) for v in losses.values())
    assert all(v.dtype == torch.float32 for v in losses.values())

    # Check keys
    assert set(losses.keys()) == {"loss"}

    # Check shapes - loss should be scalar tensor
    assert all(v.ndim == 0 for v in losses.values())

    # Check values are non-negative
    assert all(v.item() >= 0 for v in losses.values())


def test_sic_loss_shapetype_multi():
    """Test that the loss function outputs have correct shapes and types for multi-class classification."""
    model = make_model(num_classes=5, pretrained=False)
    batch_size = 4
    image = torch.randn(batch_size, 3, 224, 224)
    outputs = model(image)
    
    # Create ground truth targets for multi-class classification
    targets = torch.randint(0, 5, (batch_size,))  # Class indices
    
    # Call loss function
    losses = model.loss(outputs, targets)

    # Check types
    assert isinstance(losses, dict)
    assert all(isinstance(v, torch.Tensor) for v in losses.values())
    assert all(v.dtype == torch.float32 for v in losses.values())

    # Check keys
    assert set(losses.keys()) == {"loss"}

    # Check shapes - loss should be scalar tensor
    assert all(v.ndim == 0 for v in losses.values())

    # Check values are non-negative
    assert all(v.item() >= 0 for v in losses.values())


def test_sic_postprocess_shapetype_binary():
    """Test that the postprocess method outputs have correct shapes and types for binary classification."""
    model = make_model(num_classes=2, pretrained=False)
    batch_size = 3
    image = torch.randn(batch_size, 3, 224, 224)
    outputs = model(image)
    
    # Call postprocess function
    confidences = model.postprocess(outputs)

    # Check types
    assert isinstance(confidences, torch.Tensor)
    assert confidences.dtype == torch.float32

    # Check shapes
    assert confidences.shape[0] == batch_size
    assert confidences.shape[1] == 2  # Binary classification outputs 2 probabilities

    # Check values are in [0,1] range (after sigmoid)
    assert torch.all(confidences >= 0.0)
    assert torch.all(confidences <= 1.0)


def test_sic_postprocess_shapetype_multi():
    """Test that the postprocess method outputs have correct shapes and types for multi-class classification."""
    model = make_model(num_classes=5, pretrained=False)
    batch_size = 4
    image = torch.randn(batch_size, 3, 224, 224)
    outputs = model(image)
    
    # Call postprocess function
    confidences = model.postprocess(outputs)

    # Check types
    assert isinstance(confidences, torch.Tensor)
    assert confidences.dtype == torch.float32

    # Check shapes
    assert confidences.shape[0] == batch_size
    assert confidences.shape[1] == 5  # Multi-class classification outputs 5 probabilities

    # Check values are in [0,1] range (after softmax)
    assert torch.all(confidences >= 0.0)
    assert torch.all(confidences <= 1.0)

    # Check that probabilities sum to 1 for each sample (softmax property)
    prob_sums = torch.sum(confidences, dim=1)
    assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-6)


def test_make_model_custom_configurations():
    """Test that make_model works with custom configurations."""
    # Test explicit BCE loss
    model = make_model(num_classes=2, loss_type="bce", pretrained=False)
    assert isinstance(model._loss_function, torch.nn.BCEWithLogitsLoss)
    
    # Test explicit CrossEntropy loss
    model = make_model(num_classes=5, loss_type="ce", pretrained=False)
    assert isinstance(model._loss_function, torch.nn.CrossEntropyLoss)
    
    # Test explicit sigmoid postprocess
    model = make_model(num_classes=2, postprocess_type="sigmoid", pretrained=False)
    assert model._postprocess_function == torch.sigmoid
    
    # Test explicit softmax postprocess
    model = make_model(num_classes=5, postprocess_type="softmax", pretrained=False)
    assert model._postprocess_function.func == torch.nn.functional.softmax


def test_make_model_invalid_configurations():
    """Test that make_model raises errors for invalid configurations."""
    # Test invalid backbone type
    with pytest.raises(ValueError, match="Unknown backbone type"):
        make_model(num_classes=2, backbone_type="invalid", pretrained=False)
    
    # Test invalid head type
    with pytest.raises(ValueError, match="Unknown head type"):
        make_model(num_classes=2, head_type="invalid", pretrained=False)
    
    # Test invalid loss type
    with pytest.raises(ValueError, match="Unknown loss type"):
        make_model(num_classes=2, loss_type="invalid", pretrained=False)
    
    # Test invalid postprocess type
    with pytest.raises(ValueError, match="Unknown postprocess type"):
        make_model(num_classes=2, postprocess_type="invalid", pretrained=False)


def test_sic_composition_components():
    """Test that SIC correctly uses composed components."""
    # Create custom components
    backbone = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
    )
    predictions_head = torch.nn.Linear(64, 3)
    loss_function = torch.nn.CrossEntropyLoss()
    postprocess_function = lambda x: torch.nn.functional.softmax(x, dim=1)
    
    # Create SIC with custom components
    model = SIC(
        backbone=backbone,
        predictions_head=predictions_head,
        loss_function=loss_function,
        postprocess_function=postprocess_function,
    )
    
    # Test forward pass
    batch_size = 2
    image = torch.randn(batch_size, 3, 32, 32)
    outputs = model(image)
    
    # Check shapes
    assert outputs.shape == (batch_size, 3)
    
    # Test loss
    targets = torch.randint(0, 3, (batch_size,))
    losses = model.loss(outputs, targets)
    assert "loss" in losses
    assert losses["loss"].item() >= 0
    
    # Test postprocess
    confidences = model.postprocess(outputs)
    assert confidences.shape == (batch_size, 3)
    assert torch.all(confidences >= 0.0)
    assert torch.all(confidences <= 1.0)
