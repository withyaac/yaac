"""Tests for loading SIC models from exported checkpoints."""

import json
from pathlib import Path

import pytest
import torch

from yaac.common.model_loader import load_model_from_checkpoint
from yaac.models.sic.sic import SIC


def test_load_resnet18_model_from_checkpoint(tmp_path: Path) -> None:
    """Test loading a ResNet18 SIC model from checkpoint.
    
    This test requires a checkpoint to be created first using:
    python -m yaac_internal.scripts.create_sic_model_local
    
    Args:
        tmp_path: Temporary directory (pytest fixture)
    """
    # Check for checkpoint in common location or relative to test file
    checkpoint_dir = Path("/home/phil/data/temp/sic_models/resnet18")
    
    # Fallback to relative path if default doesn't exist
    if not checkpoint_dir.exists():
        checkpoint_dir = Path(__file__).parent / "test_checkpoints" / "resnet18"
    
    if not checkpoint_dir.exists():
        pytest.skip(
            f"Checkpoint not found at {checkpoint_dir}. "
            "Create it using create_sic_model_local.py script."
        )
    
    # Load model
    model, config = load_model_from_checkpoint(checkpoint_dir, device="cpu")
    
    # Verify model type
    assert isinstance(model, SIC)
    
    # Verify config
    assert config["model_type"] == "sic"
    assert config["backbone_type"] == "resnet18"
    assert config["num_classes"] == 2
    assert "classes" in config
    assert len(config["classes"]) == 2
    
    # Test forward pass
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    
    # Verify output shape
    # For binary classification (num_classes=2), we output 1 logit (not 2)
    # This matches yaac_internal's behavior: binary classification uses BCEWithLogitsLoss
    # with a single logit output (sigmoid activation)
    assert output.shape == (1, 1)  # (batch, 1 logit for binary classification)


def test_load_convnext_model_from_checkpoint(tmp_path: Path) -> None:
    """Test loading a ConvNeXt-Tiny DINOv3 SIC model from checkpoint.
    
    This test requires a checkpoint to be created first using:
    python -m yaac_internal.scripts.create_sic_model_local
    
    Args:
        tmp_path: Temporary directory (pytest fixture)
    """
    # Check for checkpoint in common location or relative to test file
    checkpoint_dir = Path("/home/phil/data/temp/sic_models/convnext-tiny-dinov3")
    
    # Fallback to relative path if default doesn't exist
    if not checkpoint_dir.exists():
        checkpoint_dir = Path(__file__).parent / "test_checkpoints" / "convnext_tiny_dinov3"
    
    if not checkpoint_dir.exists():
        pytest.skip(
            f"Checkpoint not found at {checkpoint_dir}. "
            "Create it using create_sic_model_local.py script."
        )
    
    # Load model
    model, config = load_model_from_checkpoint(checkpoint_dir, device="cpu")
    
    # Verify model type
    assert isinstance(model, SIC)
    
    # Verify config
    assert config["model_type"] == "sic"
    assert config["backbone_type"] == "convnext_tiny_dinov3"
    assert config["num_classes"] == 2
    assert "classes" in config
    assert len(config["classes"]) == 2
    
    # Test forward pass
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    
    # Verify output shape
    # For binary classification (num_classes=2), we output 1 logit (not 2)
    # This matches yaac_internal's behavior: binary classification uses BCEWithLogitsLoss
    # with a single logit output (sigmoid activation)
    assert output.shape == (1, 1)  # (batch, 1 logit for binary classification)


def test_load_model_missing_files(tmp_path: Path) -> None:
    """Test that loading fails gracefully when files are missing.
    
    Args:
        tmp_path: Temporary directory (pytest fixture)
    """
    # Create empty directory
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    # Should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_model_from_checkpoint(empty_dir, device="cpu")


def test_load_model_invalid_config(tmp_path: Path) -> None:
    """Test that loading fails gracefully with invalid config.
    
    Args:
        tmp_path: Temporary directory (pytest fixture)
    """
    # Create directory with invalid config
    invalid_dir = tmp_path / "invalid"
    invalid_dir.mkdir()
    
    # Create safetensors file (empty, but exists)
    (invalid_dir / "model.safetensors").touch()
    
    # Create invalid config
    invalid_config = {"model_type": "unknown_model"}
    with open(invalid_dir / "config.json", "w") as f:
        json.dump(invalid_config, f)
    
    # Should raise ValueError for unsupported model type
    with pytest.raises(ValueError, match="Unsupported model_type"):
        load_model_from_checkpoint(invalid_dir, device="cpu")
