#!/usr/bin/env python
"""
Project Verification Script
Run this to verify your installation and setup
"""

import sys
import os
from pathlib import Path

print("\n" + "="*70)
print("🔍 LUNG CANCER DETECTION SYSTEM - VERIFICATION SCRIPT")
print("="*70 + "\n")

# Step 1: Check Python version
print("✓ Checking Python version...")
python_version = sys.version_info
if python_version.major >= 3 and python_version.minor >= 8:
    print(f"  ✅ Python {python_version.major}.{python_version.minor} (OK)")
else:
    print(f"  ❌ Python {python_version.major}.{python_version.minor} (requires 3.8+)")
    sys.exit(1)

# Step 2: Check project structure
print("\n✓ Checking project structure...")
required_dirs = ["src", "api", "models"]
required_files = [
    "src/config.py", "src/model.py", "src/preprocessing.py",
    "src/inference.py", "src/train.py", "src/utils.py",
    "api/main.py", "api/routes.py", "api/schemas.py",
    "requirements.txt", "README.md"
]

all_exist = True
for dir_name in required_dirs:
    if Path(dir_name).exists():
        print(f"  ✅ {dir_name}/")
    else:
        print(f"  ❌ {dir_name}/ (MISSING)")
        all_exist = False

for file_name in required_files:
    if Path(file_name).exists():
        print(f"  ✅ {file_name}")
    else:
        print(f"  ❌ {file_name} (MISSING)")
        all_exist = False

if not all_exist:
    print("\n⚠️  Some files are missing!")
    sys.exit(1)

# Step 3: Check dependencies
print("\n✓ Checking Python dependencies...")
required_packages = [
    "torch", "torchvision", "fastapi", "uvicorn",
    "cv2", "numpy", "PIL", "pydantic"
]

missing_packages = []
for package in required_packages:
    try:
        if package == "cv2":
            import cv2
            print(f"  ✅ opencv-python")
        elif package == "PIL":
            from PIL import Image
            print(f"  ✅ Pillow")
        else:
            __import__(package)
            print(f"  ✅ {package}")
    except ImportError:
        print(f"  ❌ {package} (NOT INSTALLED)")
        missing_packages.append(package)

if missing_packages:
    print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

# Step 4: Check GPU availability (optional)
print("\n✓ Checking GPU availability...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"  ✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"     CUDA: {torch.version.cuda}")
        print(f"     cuDNN: {torch.backends.cudnn.version()}")
    else:
        print(f"  ⚠️  GPU not available (CPU mode will be slower)")
except Exception as e:
    print(f"  ⚠️  Could not check GPU: {str(e)}")

# Step 5: Test import of core modules
print("\n✓ Testing core module imports...")
try:
    sys.path.insert(0, "src")
    
    from config import MODEL_NAME, NUM_CLASSES, IMAGE_SIZE
    print(f"  ✅ config.py (model={MODEL_NAME}, classes={NUM_CLASSES})")
    
    from utils import setup_logger
    print(f"  ✅ utils.py")
    
    from preprocessing import create_preprocessor
    print(f"  ✅ preprocessing.py")
    
    from model import create_model
    print(f"  ✅ model.py")
    
    from inference import create_predictor
    print(f"  ✅ inference.py")
    
    from train import Trainer
    print(f"  ✅ train.py")
    
    from api.schemas import PredictionResponse
    print(f"  ✅ api/schemas.py")
    
except Exception as e:
    print(f"  ❌ Import failed: {str(e)}")
    sys.exit(1)

# Step 6: Test model creation
print("\n✓ Testing model creation...")
try:
    model = create_model(
        model_name="efficientnet_b0",
        num_classes=4,
        pretrained=False  # Don't download to avoid network
    )
    total_params = model.get_total_params()
    trainable_params = model.get_trainable_params()
    print(f"  ✅ Model created")
    print(f"     Total params: {total_params:,}")
    print(f"     Trainable params: {trainable_params:,}")
except Exception as e:
    print(f"  ❌ Model creation failed: {str(e)}")
    sys.exit(1)

# Step 7: Test preprocessing
print("\n✓ Testing preprocessing pipeline...")
try:
    import numpy as np
    
    preprocessor = create_preprocessor()
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    tensor, enhanced = preprocessor.preprocess(test_image)
    
    print(f"  ✅ Preprocessing pipeline")
    print(f"     Input shape: {test_image.shape}")
    print(f"     Tensor shape: {tensor.shape}")
    print(f"     Enhanced shape: {enhanced.shape}")
    print(f"     Tensor range: [{tensor.min():.2f}, {tensor.max():.2f}]")
except Exception as e:
    print(f"  ❌ Preprocessing failed: {str(e)}")
    sys.exit(1)

# Step 8: Check FastAPI
print("\n✓ Checking FastAPI setup...")
try:
    from fastapi import FastAPI
    from api.routes import router
    print(f"  ✅ FastAPI and routes imported")
except Exception as e:
    print(f"  ❌ FastAPI setup failed: {str(e)}")
    sys.exit(1)

# Summary
print("\n" + "="*70)
print("✅ ALL CHECKS PASSED!")
print("="*70)

print("\n📋 Next Steps:")
print("  1. Prepare your dataset:")
print("     - Create: data/stage_1/, data/stage_2/, data/stage_3/, data/stage_4/")
print("     - Add CT scan images (~100+ per class)")
print()
print("  2. Train the model:")
print("     python train_standalone.py --data-dir data --num-epochs 50")
print()
print("  3. Start the API:")
print("     python api/main.py")
print()
print("  4. Make predictions:")
print("     curl -X POST 'http://localhost:8000/api/predict' \\")
print("       -F 'file=@ct_scan.jpg'")
print()
print("📚 Documentation:")
print("  - README.md - Complete guide")
print("  - QUICKSTART.md - 5-minute setup")
print("  - ARCHITECTURE.md - Technical details")
print("  - TESTING.md - Testing guide")
print()
print("🚀 System is ready!")

print("\n")
