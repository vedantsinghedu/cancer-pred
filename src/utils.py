"""
Utility functions for the Lung Cancer Detection System.
Contains helper functions for image handling, logging, and validation.
"""

import io
import base64
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image
import torch

from config import CLASS_NAMES, DEVICE


def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Set up a logger with console and optional file handler.
    
    Args:
        name: Logger name
        log_file: Optional file path for logging
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array (BGR)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return image


def image_to_base64(image: np.ndarray) -> str:
    """
    Convert OpenCV image to base64 string.
    
    Args:
        image: Image as numpy array
        
    Returns:
        Base64 encoded string
    """
    _, buffer = cv2.imencode('.jpg', image)
    image_bytes = buffer.tobytes()
    return base64.b64encode(image_bytes).decode('utf-8')


def base64_to_image(b64_string: str) -> np.ndarray:
    """
    Convert base64 string to OpenCV image.
    
    Args:
        b64_string: Base64 encoded image string
        
    Returns:
        Image as numpy array
    """
    image_bytes = base64.b64decode(b64_string)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to specified size.
    
    Args:
        image: Image as numpy array
        size: Target size (height, width)
        
    Returns:
        Resized image
    """
    return cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)


def get_device() -> torch.device:
    """
    Get the appropriate device (cuda or cpu).
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available() and DEVICE == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


def get_class_name(class_idx: int) -> str:
    """
    Get class name from index.
    
    Args:
        class_idx: Class index
        
    Returns:
        Class name
    """
    return CLASS_NAMES.get(class_idx, "Unknown")


def format_prediction(
    class_idx: int,
    confidence: float,
    probabilities: Dict[int, float],
    enhanced_image: np.ndarray = None
) -> Dict[str, Any]:
    """
    Format prediction results into a dictionary.
    
    Args:
        class_idx: Predicted class index
        confidence: Confidence score
        probabilities: Dictionary of class probabilities
        enhanced_image: Optional enhanced image array
        
    Returns:
        Formatted prediction dictionary
    """
    result = {
        "stage": get_class_name(class_idx),
        "confidence": float(confidence),
        "probabilities": {
            get_class_name(idx): float(prob)
            for idx, prob in probabilities.items()
        },
    }
    
    if enhanced_image is not None:
        result["enhanced_image"] = image_to_base64(enhanced_image)
    
    return result


def create_data_loaders(
    dataset,
    batch_size: int,
    train_split: float = 0.8,
    val_split: float = 0.1,
    num_workers: int = 4
) -> Tuple:
    """
    Create PyTorch DataLoaders for training, validation, and test sets.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        train_split: Training split ratio
        val_split: Validation split ratio
        num_workers: Number of worker threads
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import random_split, DataLoader
    
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
