"""
Model architecture and loading module.
Implements transfer learning with EfficientNet or ResNet.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torchvision import models

from config import (
    MODEL_NAME,
    NUM_CLASSES,
    PRETRAINED,
    DEVICE,
)
from utils import get_device, setup_logger

logger = setup_logger(__name__)


class LungCancerModel(nn.Module):
    """
    Lung cancer classification model using transfer learning.
    Uses pretrained EfficientNet or ResNet backbone with custom classifier head.
    """
    
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = PRETRAINED,
        freeze_backbone: bool = False,
    ):
        """
        Initialize model.
        
        Args:
            model_name: Model architecture (efficientnet_b0, resnet50, etc.)
            num_classes: Number of output classes
            pretrained: Load pretrained weights
            freeze_backbone: Freeze backbone layers for inference
        """
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pretrained backbone
        if "efficientnet" in model_name.lower():
            self.backbone = self._load_efficientnet(model_name, pretrained)
            backbone_features = self.backbone.classifier[1].in_features
        elif "resnet" in model_name.lower():
            self.backbone = self._load_resnet(model_name, pretrained)
            backbone_features = self.backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Replace classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(backbone_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
        
        if freeze_backbone:
            self._freeze_backbone()
        
        logger.info(f"Model initialized: {model_name} with {num_classes} classes")
    
    @staticmethod
    def _load_efficientnet(model_name: str, pretrained: bool) -> nn.Module:
        """Load EfficientNet model."""
        if model_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
        elif model_name == "efficientnet_b1":
            model = models.efficientnet_b1(weights='DEFAULT' if pretrained else None)
        elif model_name == "efficientnet_b2":
            model = models.efficientnet_b2(weights='DEFAULT' if pretrained else None)
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {model_name}")
        return model
    
    @staticmethod
    def _load_resnet(model_name: str, pretrained: bool) -> nn.Module:
        """Load ResNet model."""
        if model_name == "resnet50":
            model = models.resnet50(weights='DEFAULT' if pretrained else None)
        elif model_name == "resnet101":
            model = models.resnet101(weights='DEFAULT' if pretrained else None)
        elif model_name == "resnet34":
            model = models.resnet34(weights='DEFAULT' if pretrained else None)
        else:
            raise ValueError(f"Unsupported ResNet variant: {model_name}")
        return model
    
    def _freeze_backbone(self):
        """Freeze backbone layers."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone layers frozen")
    
    def _unfreeze_backbone(self, num_layers: int = 2):
        """
        Unfreeze last few layers of backbone for fine-tuning.
        
        Args:
            num_layers: Number of last layers to unfreeze
        """
        # Unfreeze last num_layers
        for param in list(self.backbone.parameters())[-num_layers:]:
            param.requires_grad = True
        logger.info(f"Unfroze last {num_layers} layers of backbone")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, 3, height, width)
            
        Returns:
            Class logits (batch_size, num_classes)
        """
        # Extract features from backbone
        if isinstance(self.backbone, models.EfficientNet):
            features = self.backbone.features(x)
            features = self.backbone.avgpool(features)
            features = torch.flatten(features, 1)
        else:  # ResNet
            features = self.backbone(x)
        
        # Classify
        logits = self.classifier(features)
        return logits
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions with probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (class indices, probabilities)
        """
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        class_idx = torch.argmax(probs, dim=1)
        return class_idx, probs
    
    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_model(
    model_name: str = MODEL_NAME,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = PRETRAINED,
    freeze_backbone: bool = False,
) -> LungCancerModel:
    """
    Factory function to create model.
    
    Args:
        model_name: Model architecture
        num_classes: Number of classes
        pretrained: Use pretrained weights
        freeze_backbone: Freeze backbone for inference
        
    Returns:
        Model instance
    """
    return LungCancerModel(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )


def load_model(
    checkpoint_path: str,
    model_name: str = MODEL_NAME,
    num_classes: int = NUM_CLASSES,
    device: str = DEVICE,
) -> Tuple[LungCancerModel, dict]:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to saved model
        model_name: Model architecture
        num_classes: Number of classes
        device: Device to load on
        
    Returns:
        Tuple of (model, checkpoint_dict)
    """
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(get_device())
    model.eval()
    
    logger.info(f"Model loaded from {checkpoint_path}")
    return model, checkpoint


def save_model(
    model: LungCancerModel,
    optimizer,
    epoch: int,
    loss: float,
    checkpoint_path: str,
    is_best: bool = False,
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        checkpoint_path: Path to save checkpoint
        is_best: Whether this is the best model
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_name': model.model_name,
        'num_classes': model.num_classes,
    }
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    if is_best:
        logger.info("New best model found!")
