"""
Inference module for making predictions on lung CT scans.
Combines preprocessing and model prediction into a unified pipeline.
"""

from pathlib import Path
from typing import Dict, Tuple, Union

import cv2
import numpy as np
import torch

from preprocessing import create_preprocessor
from model import load_model, LungCancerModel
from utils import get_device, setup_logger, format_prediction, get_class_name

logger = setup_logger(__name__)


class LungCancerPredictor:
    """
    End-to-end inference pipeline for lung cancer stage prediction.
    Handles preprocessing, model loading, and prediction formatting.
    """
    
    def __init__(
        self,
        model_checkpoint: Union[str, Path],
        model_name: str = "efficientnet_b0",
        num_classes: int = 4,
        image_size: int = 224,
        device: str = "cuda",
    ):
        """
        Initialize predictor.
        
        Args:
            model_checkpoint: Path to saved model checkpoint
            model_name: Model architecture name
            num_classes: Number of output classes
            image_size: Target image size
            device: Device to use (cuda or cpu)
        """
        self.device = get_device() if device == "cuda" else torch.device("cpu")
        self.preprocessor = create_preprocessor(image_size=image_size)
        
        # Load model
        self.model, self.checkpoint = load_model(
            checkpoint_path=str(model_checkpoint),
            model_name=model_name,
            num_classes=num_classes,
            device=str(self.device),
        )
        self.num_classes = num_classes
        
        logger.info(
            f"Predictor initialized with {model_name} "
            f"on device {self.device}"
        )
    
    def predict_from_file(self, image_path: Union[str, Path]) -> Dict:
        """
        Predict cancer stage from image file.
        
        Args:
            image_path: Path to CT scan image
            
        Returns:
            Dictionary with prediction results
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        return self.predict_from_array(image)
    
    def predict_from_array(self, image: np.ndarray) -> Dict:
        """
        Predict cancer stage from image array.
        
        Args:
            image: Image as numpy array (BGR from cv2)
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        tensor, enhanced_image = self.preprocessor.preprocess(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            class_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, class_idx].item()
        
        # Format probabilities
        class_probs = {
            i: probs[0, i].item() for i in range(self.num_classes)
        }
        
        # Format results
        result = format_prediction(
            class_idx=class_idx,
            confidence=confidence,
            probabilities=class_probs,
            enhanced_image=enhanced_image,
        )
        
        logger.info(
            f"Prediction: {result['stage']} "
            f"(confidence: {confidence:.4f})"
        )
        
        return result
    
    def predict_batch(
        self,
        images: list,
        batch_size: int = 8,
    ) -> list:
        """
        Predict on multiple images.
        
        Args:
            images: List of image arrays
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Preprocess batch
            tensors = []
            enhanced_images = []
            
            for img in batch:
                tensor, enhanced = self.preprocessor.preprocess(img)
                tensors.append(tensor)
                enhanced_images.append(enhanced)
            
            # Stack into batch
            batch_tensor = torch.stack(tensors).to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.model(batch_tensor)
                probs = torch.softmax(logits, dim=1)
                class_indices = torch.argmax(probs, dim=1)
            
            # Format results
            for j, class_idx in enumerate(class_indices):
                idx = class_idx.item()
                confidence = probs[j, idx].item()
                class_probs = {
                    k: probs[j, k].item() for k in range(self.num_classes)
                }
                
                result = format_prediction(
                    class_idx=idx,
                    confidence=confidence,
                    probabilities=class_probs,
                    enhanced_image=enhanced_images[j],
                )
                results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model.model_name,
            "num_classes": self.num_classes,
            "class_names": {i: get_class_name(i) for i in range(self.num_classes)},
            "total_parameters": self.model.get_total_params(),
            "trainable_parameters": self.model.get_trainable_params(),
            "device": str(self.device),
            "checkpoint_epoch": self.checkpoint.get("epoch", "unknown"),
        }


def create_predictor(
    model_checkpoint: Union[str, Path],
    model_name: str = "efficientnet_b0",
    num_classes: int = 4,
    image_size: int = 224,
    device: str = "cuda",
) -> LungCancerPredictor:
    """
    Factory function to create predictor.
    
    Args:
        model_checkpoint: Path to model checkpoint
        model_name: Model architecture
        num_classes: Number of classes
        image_size: Target image size
        device: Device to use
        
    Returns:
        Predictor instance
    """
    return LungCancerPredictor(
        model_checkpoint=model_checkpoint,
        model_name=model_name,
        num_classes=num_classes,
        image_size=image_size,
        device=device,
    )
