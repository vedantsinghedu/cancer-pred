"""
Image preprocessing and enhancement module.
Implements CLAHE, filtering, and normalization for lung CT scans.
"""

from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from config import (
    IMAGE_SIZE,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_SIZE,
    FILTER_KERNEL_SIZE,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
)


class ImagePreprocessor:
    """
    Handles image preprocessing and enhancement for lung CT scans.
    Uses CLAHE, denoising, and normalization for fast, lightweight processing.
    """
    
    def __init__(
        self,
        image_size: int = IMAGE_SIZE,
        clahe_clip_limit: float = CLAHE_CLIP_LIMIT,
        clahe_tile_size: int = CLAHE_TILE_SIZE,
        filter_kernel_size: int = FILTER_KERNEL_SIZE,
    ):
        """
        Initialize preprocessor.
        
        Args:
            image_size: Target image size
            clahe_clip_limit: CLAHE contrast limit
            clahe_tile_size: CLAHE tile grid size
            filter_kernel_size: Denoising filter kernel size
        """
        self.image_size = image_size
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_size, clahe_tile_size)
        )
        self.filter_kernel_size = filter_kernel_size
        
        # PyTorch transforms for normalization
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=NORMALIZATION_MEAN,
                std=NORMALIZATION_STD
            ),
        ])
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization.
        
        Args:
            image: Input image (should be grayscale or converted)
            
        Returns:
            CLAHE processed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return self.clahe.apply(image)
    
    def apply_denoising(self, image: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filtering for denoising while preserving edges.
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        # Apply bilateral filter (better edge preservation than Gaussian)
        denoised = cv2.bilateralFilter(
            image,
            d=self.filter_kernel_size,
            sigmaColor=75,
            sigmaSpace=75
        )
        return denoised
    
    def resize(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        return cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR
        )
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values to [0, 1] range.
        
        Args:
            image: Input image (uint8)
            
        Returns:
            Normalized image (float32 in [0, 1])
        """
        return image.astype(np.float32) / 255.0
    
    def preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Full preprocessing pipeline.
        1. Convert to grayscale
        2. Apply CLAHE
        3. Apply bilateral filtering
        4. Resize
        5. Normalize
        6. Convert to tensor
        
        Args:
            image: Input image (BGR format from cv2)
            
        Returns:
            Tuple of (tensor for model, enhanced_image for visualization)
        """
        # Step 1: Convert BGR to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Step 2: Apply CLAHE for contrast enhancement
        enhanced = self.apply_clahe(gray)
        
        # Step 3: Apply bilateral filter for denoising
        denoised = self.apply_denoising(enhanced)
        
        # Step 4: Resize to target size
        resized = self.resize(denoised)
        
        # Step 5: Normalize
        normalized = self.normalize(resized)
        
        # Convert to grayscale 3-channel for model compatibility
        rgb_image = np.stack([normalized] * 3, axis=-1)
        
        # Step 6: Convert to tensor with ImageNet normalization
        tensor = self.transforms(rgb_image)
        
        # Prepare enhanced image for output (convert back to uint8)
        enhanced_for_output = (normalized * 255).astype(np.uint8)
        enhanced_for_output = cv2.cvtColor(enhanced_for_output, cv2.COLOR_GRAY2BGR)
        
        return tensor, enhanced_for_output


def create_preprocessor(image_size: int = IMAGE_SIZE) -> ImagePreprocessor:
    """
    Factory function to create ImagePreprocessor instance.
    
    Args:
        image_size: Target image size
        
    Returns:
        ImagePreprocessor instance
    """
    return ImagePreprocessor(image_size=image_size)
