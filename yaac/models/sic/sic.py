import torchvision
import torch
from yaac.common.trainable_model import TrainableModel
from functools import partial


class SIC(TrainableModel):
    """Simple Image Classifier using composition pattern.
    
    This class uses composition to separate concerns between the backbone,
    predictions head, loss function, and postprocessing function.
    """
    
    def __init__(
        self,
        backbone: torch.nn.Module,
        predictions_head: torch.nn.Module,
        loss_function: torch.nn.Module,
        postprocess_function: callable,
    ):
        """Initialize SIC with composed components.
        
        Args:
            backbone: Feature extraction backbone network
            predictions_head: Final classification head
            loss_function: Loss function for training
            postprocess_function: Function to postprocess raw outputs
        """
        super().__init__()
        self._backbone = backbone
        self._predictions_head = predictions_head
        self._loss_function = loss_function
        self._postprocess_function = postprocess_function

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            image: Input image tensor
            
        Returns:
            Raw logits from the predictions head
        """
        features = self._backbone(image)
        logits = self._predictions_head(features)
        return logits

    def loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute loss between outputs and targets.
        
        Args:
            outputs: Raw model outputs
            targets: Ground truth targets
            
        Returns:
            Dictionary containing the loss value
        """
        loss = self._loss_function(outputs, targets)
        return {"loss": loss}

    def postprocess(self, outputs: torch.Tensor) -> torch.Tensor:
        """Postprocess raw outputs into final predictions.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Processed predictions (probabilities or confidences)
        """
        confidences = self._postprocess_function(outputs)
        return confidences


def make_model(
    num_classes: int,
    backbone_type: str = "resnet18",
    head_type: str = "basic_001",
    loss_type: str = "auto",
    postprocess_type: str = "auto",
    pretrained: bool = True,
) -> SIC:
    """Factory function to create a SIC model with configurable components.
    
    This function creates a SIC model using configurable backbone, head,
    loss function, and postprocessing components.
    
    Args:
        num_classes: Number of output classes
        backbone_type: Type of backbone to use
        head_type: Type of predictions head to use
        loss_type: Type of loss function to use ("auto", "bce", "ce")
        postprocess_type: Type of postprocess function to use ("auto", "sigmoid", "softmax")
        pretrained: Whether to use pretrained weights for the backbone
        
    Returns:
        Configured SIC model instance
    """
    backbone = _build_backbone(backbone_type, pretrained)
    predictions_head = _build_head(head_type, num_classes)
    loss_function = _build_loss_function(loss_type, num_classes)
    postprocess_function = _build_postprocess_function(postprocess_type, num_classes)
    
    return SIC(
        backbone=backbone,
        predictions_head=predictions_head,
        loss_function=loss_function,
        postprocess_function=postprocess_function,
    )


def _build_backbone(backbone_type: str, pretrained: bool) -> torch.nn.Module:
    """Build backbone network based on type.
    
    Args:
        backbone_type: Type of backbone to build
        pretrained: Whether to use pretrained weights
        
    Returns:
        Backbone network module
    """
    if backbone_type == "resnet18":
        if pretrained:
            backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            backbone = torchvision.models.resnet18(weights=None)
        # Remove avgpool and fc layers, keep only the feature extraction part
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        return backbone
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")


def _build_head(head_type: str, num_classes: int) -> torch.nn.Module:
    """Build predictions head based on type.
    
    Args:
        head_type: Type of head to build
        num_classes: Number of output classes
        
    Returns:
        Predictions head module
    """
    if head_type == "basic_001":
        # Basic head with adaptive pooling and linear layer
        # Assumes ResNet18 feature dimension is 512
        return torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(512, num_classes)
        )
    else:
        raise ValueError(f"Unknown head type: {head_type}")


def _build_loss_function(loss_type: str, num_classes: int) -> torch.nn.Module:
    """Build loss function based on type and number of classes.
    
    Args:
        loss_type: Type of loss function to build
        num_classes: Number of output classes
        
    Returns:
        Loss function module
    """
    if loss_type == "auto":
        # Automatically choose based on number of classes
        if num_classes == 1 or num_classes == 2:
            return torch.nn.BCEWithLogitsLoss()
        else:
            return torch.nn.CrossEntropyLoss()
    elif loss_type == "bce":
        return torch.nn.BCEWithLogitsLoss()
    elif loss_type == "ce":
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def _build_postprocess_function(postprocess_type: str, num_classes: int) -> callable:
    """Build postprocess function based on type and number of classes.
    
    Args:
        postprocess_type: Type of postprocess function to build
        num_classes: Number of output classes
        
    Returns:
        Postprocess function
    """
    if postprocess_type == "auto":
        # Automatically choose based on number of classes
        if num_classes == 1 or num_classes == 2:
            return torch.sigmoid
        else:
            return partial(torch.nn.functional.softmax, dim=1)
    elif postprocess_type == "sigmoid":
        return torch.sigmoid
    elif postprocess_type == "softmax":
        return partial(torch.nn.functional.softmax, dim=1)
    else:
        raise ValueError(f"Unknown postprocess type: {postprocess_type}")


if __name__ == "__main__":
    model = make_model(num_classes=10)
    image = torch.randn(1, 3, 224, 224)
    outputs = model(image)
    breakpoint()
    abc = 1