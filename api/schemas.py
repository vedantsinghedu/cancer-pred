"""
Pydantic schemas for FastAPI request/response validation.
Defines data models for API endpoints.
"""

from typing import Dict, Optional

from pydantic import BaseModel, Field


class HealthCheckResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    message: str = Field(..., description="Status message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "message": "Service is running",
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response schema."""
    stage: str = Field(..., description="Predicted cancer stage")
    confidence: float = Field(
        ...,
        description="Confidence score (0-1)",
        ge=0.0,
        le=1.0,
    )
    probabilities: Dict[str, float] = Field(
        ...,
        description="Class probabilities",
    )
    enhanced_image: Optional[str] = Field(
        None,
        description="Base64 encoded enhanced image",
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "stage": "Stage II",
                "confidence": 0.92,
                "probabilities": {
                    "Stage I": 0.05,
                    "Stage II": 0.92,
                    "Stage III": 0.02,
                    "Stage IV": 0.01,
                },
                "enhanced_image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",  # noqa
            }
        }


class ModelInfoResponse(BaseModel):
    """Model information response schema."""
    model_name: str = Field(..., description="Model architecture")
    num_classes: int = Field(..., description="Number of output classes")
    class_names: Dict[int, str] = Field(..., description="Class names mapping")
    total_parameters: int = Field(..., description="Total model parameters")
    trainable_parameters: int = Field(..., description="Trainable parameters")
    device: str = Field(..., description="Device used (cuda/cpu)")
    checkpoint_epoch: int = Field(..., description="Epoch of loaded checkpoint")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "efficientnet_b0",
                "num_classes": 4,
                "class_names": {
                    "0": "Stage I",
                    "1": "Stage II",
                    "2": "Stage III",
                    "3": "Stage IV",
                },
                "total_parameters": 4234321,
                "trainable_parameters": 523456,
                "device": "cuda",
                "checkpoint_epoch": 42,
            }
        }


class BatchPredictionResponse(BaseModel):
    """Batch prediction response schema."""
    total_images: int = Field(..., description="Total images processed")
    predictions: list = Field(..., description="List of predictions")
    processing_time: float = Field(..., description="Processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_images": 5,
                "predictions": [
                    {
                        "stage": "Stage II",
                        "confidence": 0.92,
                        "probabilities": {
                            "Stage I": 0.05,
                            "Stage II": 0.92,
                            "Stage III": 0.02,
                            "Stage IV": 0.01,
                        },
                    }
                ],
                "processing_time": 1.23,
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Error details")
    status_code: int = Field(..., description="HTTP status code")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid image format",
                "details": "Supported formats: JPG, PNG",
                "status_code": 400,
            }
        }
