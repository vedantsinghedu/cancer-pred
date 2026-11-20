"""
FastAPI main application for Lung Cancer Detection System.
Entry point for the REST API server.
"""

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference import create_predictor
from .routes import router, set_predictor
from config import MODEL_CHECKPOINT, API_HOST, API_PORT, WORKERS

# Create FastAPI app
app = FastAPI(
    title="Lung Cancer Stage Detection API",
    description="ML-powered REST API for detecting and classifying lung cancer stages from CT scans",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api", tags=["Prediction"])


@app.on_event("startup")
async def startup_event():
    """Initialize model and predictor on startup."""
    print("=" * 60)
    print("🚀 Starting Lung Cancer Detection API...")
    print("=" * 60)
    
    try:
        # Check if model exists
        if not Path(MODEL_CHECKPOINT).exists():
            print("⚠️  Warning: Model checkpoint not found!")
            print(f"   Expected at: {MODEL_CHECKPOINT}")
            print("   Please train the model first using: python train.py")
            print("   API will start but predictions will not work.")
            print()
        else:
            # Load predictor
            print(f"📦 Loading model from: {MODEL_CHECKPOINT}")
            predictor = create_predictor(
                model_checkpoint=str(MODEL_CHECKPOINT),
                model_name="efficientnet_b0",
                num_classes=4,
                image_size=224,
                device="cuda",
            )
            
            set_predictor(predictor)
            
            # Get model info
            info = predictor.get_model_info()
            print(f"✅ Model loaded successfully!")
            print(f"   Architecture: {info['model_name']}")
            print(f"   Classes: {info['num_classes']}")
            print(f"   Parameters: {info['total_parameters']:,}")
            print(f"   Device: {info['device']}")
            print()
    
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print("   API will start but predictions will not work.")
        print()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print()
    print("=" * 60)
    print("🛑 Shutting down API...")
    print("=" * 60)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Lung Cancer Stage Detection API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
    }


if __name__ == "__main__":
    print("Starting server...")
    print(f"Host: {API_HOST}")
    print(f"Port: {API_PORT}")
    print(f"Workers: {WORKERS}")
    print()
    
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        workers=WORKERS,
        reload=False,
    )
