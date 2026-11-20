"""
Configuration module for Lung Cancer Detection System.
Centralized settings for model, training, and inference.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
API_DIR = PROJECT_ROOT / "api"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Model configuration
MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 4
IMAGE_SIZE = 224
PRETRAINED = True
DEVICE = "cuda"  # cuda or cpu

# Class mappings
CLASS_NAMES = {
    0: "Stage I",
    1: "Stage II",
    2: "Stage III",
    3: "Stage IV",
}

# Training configuration
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
WEIGHT_DECAY = 1e-5
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Preprocessing configuration
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = 8
FILTER_KERNEL_SIZE = 5
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
NORMALIZATION_STD = [0.229, 0.224, 0.225]   # ImageNet std

# Inference configuration
CONFIDENCE_THRESHOLD = 0.5
MAX_INFERENCE_TIME = 2.0  # seconds

# Model paths
MODEL_CHECKPOINT = MODELS_DIR / "lung_cancer_model.pth"
BEST_MODEL = MODELS_DIR / "best_model.pth"
TRAINING_LOG = MODELS_DIR / "training_log.txt"

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
WORKERS = 4
