"""
FastAPI routes for lung cancer detection API.
Defines endpoints for predictions and health checks.
"""

import io
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2
import numpy as np

from .schemas import (
    PredictionResponse,
    HealthCheckResponse,
    ModelInfoResponse,
    ErrorResponse,
)

# Create router
router = APIRouter()

# Global predictor instance (will be set at startup)
predictor = None


def set_predictor(pred):
    """Set the global predictor instance."""
    global predictor
    predictor = pred


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["Health"],
    summary="Health check endpoint",
)
async def health_check() -> HealthCheckResponse:
    """
    Check API health status.
    
    Returns:
        Health status information
    """
    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        message="Lung Cancer Detection API is running",
    )


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["Info"],
    summary="Get model information",
)
async def get_model_info() -> ModelInfoResponse:
    """
    Get information about the loaded model.
    
    Returns:
        Model information
        
    Raises:
        HTTPException: If model is not loaded
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = predictor.get_model_info()
    return ModelInfoResponse(**info)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict cancer stage from CT scan",
    responses={
        200: {"description": "Successful prediction"},
        400: {"description": "Invalid image format", "model": ErrorResponse},
        500: {"description": "Server error", "model": ErrorResponse},
    },
)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Predict lung cancer stage from CT scan image.
    
    Args:
        file: CT scan image file (JPG, PNG, etc.)
        
    Returns:
        Prediction results with stage, confidence, and probabilities
        
    Raises:
        HTTPException: If image is invalid or prediction fails
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Supported: {allowed_extensions}",
        )
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        # Make prediction
        start_time = time.time()
        result = predictor.predict_from_array(image)
        inference_time = time.time() - start_time
        
        # Log inference time
        print(f"Inference time: {inference_time:.4f}s")
        
        return PredictionResponse(**result)
    
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@router.post(
    "/predict-batch",
    tags=["Prediction"],
    summary="Predict on multiple images",
    responses={
        200: {"description": "Successful batch prediction"},
        400: {"description": "Invalid request", "model": ErrorResponse},
        500: {"description": "Server error", "model": ErrorResponse},
    },
)
async def predict_batch(
    files: list = File(...),
    include_images: Optional[bool] = False,
) -> dict:
    """
    Predict on multiple CT scan images (batch processing).
    
    Args:
        files: List of CT scan image files
        include_images: Whether to include enhanced images in response
        
    Returns:
        Batch prediction results with processing time
        
    Raises:
        HTTPException: If any image is invalid or prediction fails
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    try:
        images = []
        
        # Read all images
        for file in files:
            file_ext = Path(file.filename).suffix.lower()
            allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
            
            if file_ext not in allowed_extensions:
                raise ValueError(
                    f"Invalid format for {file.filename}. "
                    f"Supported: {allowed_extensions}"
                )
            
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError(f"Failed to decode {file.filename}")
            
            images.append(image)
        
        # Make batch predictions
        start_time = time.time()
        predictions = predictor.predict_batch(images, batch_size=8)
        processing_time = time.time() - start_time
        
        # Remove enhanced images if not requested
        if not include_images:
            for pred in predictions:
                pred.pop("enhanced_image", None)
        
        return {
            "total_images": len(predictions),
            "predictions": predictions,
            "processing_time": processing_time,
        }
    
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}",
        )
