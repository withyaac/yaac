"""Common model loading utilities for yaac models.

This module provides a unified interface for loading models from exported
checkpoints (safetensors + config.json).
"""

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from yaac.models.sic.sic import SIC, make_model


def load_model_from_checkpoint(
    checkpoint_dir: Path | str,
    device: str = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """Load a model from exported checkpoint directory.
    
    The checkpoint directory should contain:
    - model.safetensors: Model weights
    - config.json: Model configuration
    
    Args:
        checkpoint_dir: Directory containing model.safetensors and config.json
        device: Device to load model on
        
    Returns:
        Tuple of (loaded_model, config_dict)
        
    Raises:
        FileNotFoundError: If checkpoint files are missing
        ValueError: If model_type in config is not supported
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Check for required files
    safetensors_path = checkpoint_dir / "model.safetensors"
    config_path = checkpoint_dir / "config.json"
    
    if not safetensors_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {safetensors_path}. "
            f"Expected file: model.safetensors"
        )
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            f"Expected file: config.json"
        )
    
    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Determine model type and create model
    model_type = config.get("model_type", "").lower()
    
    if model_type == "sic":
        model = _load_sic_model(config, device)
    else:
        raise ValueError(
            f"Unsupported model_type: {model_type}. "
            f"Supported types: 'sic'"
        )
    
    # Load weights
    state_dict = load_file(str(safetensors_path))
    
    # Load weights with strict matching
    model.load_state_dict(state_dict, strict=True)
    
    model = model.to(device)
    model.eval()
    
    return model, config


def _load_sic_model(config: dict[str, Any], device: str) -> SIC:
    """Create SIC model from config.
    
    Args:
        config: Model configuration dictionary
        device: Device to create model on
        
    Returns:
        SIC model instance (with random weights, ready for loading)
    """
    backbone_type = config.get("backbone_type", "resnet18")
    num_classes = config.get("num_classes", 2)
    
    # Determine head type based on backbone
    # This matches the logic in yaac_internal
    if backbone_type == "convnext_tiny_dinov3":
        head_type = "basic_convnext_tiny"
    else:
        head_type = "basic_001"
    
    # Override if explicitly specified in config
    if "head_type" in config:
        head_type = config["head_type"]
    
    # Create model with random weights (customers load their own trained weights)
    model = make_model(
        num_classes=num_classes,
        backbone_type=backbone_type,
        head_type=head_type,
    )
    
    return model
