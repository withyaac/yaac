
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


def create_sic_model(num_classes: int, pretrained: bool = True) -> SIC:
    """Factory function to create a SIC model with ResNet18 backbone.
    
    This function creates a SIC model using ResNet18 as the backbone,
    removing the final average pooling and fully connected layers,
    and replacing them with a custom predictions head.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights for the backbone
        
    Returns:
        Configured SIC model instance
    """
    # Create ResNet18 backbone and remove final layers
    if pretrained:
        backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        backbone = torchvision.models.resnet18(weights=None)
    # Remove avgpool and fc layers, keep only the feature extraction part
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    
    # Create predictions head
    # ResNet18 feature dimension is 512
    predictions_head = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(512, num_classes)
    )
    
    # Configure loss and postprocess functions based on number of classes
    if num_classes == 1 or num_classes == 2:
        loss_function = torch.nn.BCEWithLogitsLoss()
        postprocess_function = torch.sigmoid
    else:
        loss_function = torch.nn.CrossEntropyLoss()
        postprocess_function = partial(torch.nn.functional.softmax, dim=1)
    
    return SIC(
        backbone=backbone,
        predictions_head=predictions_head,
        loss_function=loss_function,
        postprocess_function=postprocess_function,
    )





if __name__ == "__main__":
    model = create_sic_model(num_classes=10)
    image = torch.randn(1, 3, 224, 224)
    outputs = model(image)
    breakpoint()
    abc = 1