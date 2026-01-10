"""Abstract base class for trainable models.

This module defines the TrainableModel class, which serves as an abstract base class for all
trainable models in the system. It provides a standardized interface for model training,
inference, and post-processing.
"""

from abc import ABC, abstractmethod
import torch
from typing import Dict, Any


class TrainableModel(ABC, torch.nn.Module):
    """Abstract base class for trainable models.

    This class defines the interface that all trainable models must implement. It combines
    PyTorch's nn.Module with Python's ABC to create an abstract base class that enforces a
    consistent interface across different model types.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> Any:
        """Perform the forward pass of the model.

        Subclasses can override this method to accept different types of inputs. But the
        output should be directly input to the loss and postprocess methods.

        Args:
            inputs: Input data

        Returns:
            Raw model outputs (e.g., logits, bounding boxes, feature maps). The exact type
            depends on the specific model implementation.
        """
        pass

    @abstractmethod
    def loss(self, outputs: Any, targets: Any) -> Dict[str, torch.Tensor]:
        """Compute the loss between model outputs and ground truth targets.

        Subclasses can override this method to compute different losses.
        But the input should be the output of the forward() method.

        Args:
            outputs: Raw outputs from the forward() method.
            targets: Ground truth data corresponding to the inputs.

        Returns:
            Dictionary mapping loss names to their corresponding tensor values. Multiple
            losses can be returned for multi-task learning or auxiliary objectives.
        """
        pass

    @abstractmethod
    def postprocess(self, outputs: Any) -> Any:
        """Convert raw model outputs into final predictions.

        This method handles any necessary post-processing of the raw model outputs to
        produce the final predictions in a format suitable for evaluation or deployment.

        Subclasses can override this method to postprocess the outputs in different ways.
        But the input should be the output of the forward() method.

        Examples:
            - For classification: Convert logits to class probabilities or labels
            - For detection: Convert bounding box offsets to actual boxes and apply NMS
            - For segmentation: Convert logits to pixel-wise class predictions

        Args:
            outputs: Raw outputs from the forward() method.

        Returns:
            Processed predictions in the appropriate format for the task.
        """
        pass
