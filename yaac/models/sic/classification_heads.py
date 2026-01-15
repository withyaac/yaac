"""Classification head implementations for SIC models.

This module provides classification head classes that can be used with SIC models.
"""

import torch


class SingleFCClassificationHead(torch.nn.Module):
    """Single fully-connected classification head.
    
    This head performs:
    1. Adaptive average pooling to (1, 1)
    2. Flattening
    3. Linear transformation
    
    Args:
        in_features: Number of input features
        num_classes: Number of output classes
        bias: Whether to use bias (default: True)
    """
    
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        bias: bool = True,
    ) -> None:
        """Initialize SingleFCClassificationHead.
        
        Args:
            in_features: Number of input features
            num_classes: Number of output classes
            bias: Whether to use bias
        """
        super().__init__()
        
        # Create the pooling and flatten layers
        self._pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self._flatten = torch.nn.Flatten()
        
        # Create linear layer
        self._linear = torch.nn.Linear(in_features, num_classes, bias=bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the head.
        
        Args:
            features: Input features of shape (batch, channels, height, width).
            
        Returns:
            Logits of shape (batch, num_classes).
        """
        # Pool and flatten features
        pooled = self._pool(features)
        flattened = self._flatten(pooled)
        
        # Apply linear transformation
        logits = self._linear(flattened)
        return logits
