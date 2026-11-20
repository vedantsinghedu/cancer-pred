# 🫁 Lung Cancer Stage Detection System

A production-ready ML system for detecting and classifying lung cancer stages from CT scans using transfer learning and lightweight image processing.

## 📊 Overview

This system provides:
- **Fast inference** (<2 seconds per image)
- **Transfer learning** with EfficientNet-B0 pretrained on ImageNet
- **Lightweight preprocessing** using CLAHE and bilateral filtering
- **Clean, modular architecture** for production deployment
- **REST API** via FastAPI for easy integration
- **Batch processing** support for multiple images

### Supported Cancer Stages
- Stage I
- Stage II
- Stage III
- Stage IV

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU (CUDA 11.8+) for GPU acceleration (optional but recommended)
- 4GB+ RAM

### Step 1: Clone and Setup

```bash
# Navigate to project directory
cd lung-cancer-detection

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

If you encounter GPU issues:
```bash
# For CPU-only (slower)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 📁 Project Structure

```
lung-cancer-detection/
├── src/
│   ├── config.py              # Configuration and settings
│   ├── utils.py               # Utility functions
│   ├── preprocessing.py       # Image enhancement pipeline
│   ├── model.py               # Model architecture
│   ├── inference.py           # Prediction pipeline
│   ├── train.py               # Training script
│   └── __init__.py
│
├── api/
│   ├── main.py                # FastAPI application
│   ├── routes.py              # API endpoints
│   ├── schemas.py             # Pydantic models
│   └── __init__.py
│
├── models/                    # Saved model checkpoints
│   └── (models saved here)
│
├── data/                      # Dataset directory
│   ├── train/
│   │   ├── stage_1/
│   │   ├── stage_2/
│   │   ├── stage_3/
│   │   └── stage_4/
│   ├── val/
│   └── test/
│
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🚀 Quick Start

### Option 1: Use Pretrained Model (Fast)

Download a pretrained checkpoint and place it in `models/lung_cancer_model.pth`.

### Option 2: Train Your Own Model

#### Step 1: Prepare Dataset

Organize your CT scan images in this structure:

```
data/
├── stage_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── stage_2/
│   ├── image1.jpg
│   └── ...
├── stage_3/
│   └── ...
└── stage_4/
    └── ...
```

#### Step 2: Run Training

```bash
python src/train.py \
    --data-dir data \
    --batch-size 32 \
    --num-epochs 50 \
    --learning-rate 1e-4 \
    --model-name efficientnet_b0
```

**Training outputs:**
- `models/best_model.pth` - Best checkpoint
- `models/epoch_*.pth` - Periodic checkpoints
- `models/training_log.txt` - Training logs

---

## 🔮 Inference

### Method 1: Python Script

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from inference import create_predictor

# Initialize predictor
predictor = create_predictor(
    model_checkpoint="models/best_model.pth",
    model_name="efficientnet_b0",
    num_classes=4,
    image_size=224,
    device="cuda",
)

# Predict on single image
result = predictor.predict_from_file("path/to/ct_scan.jpg")

print(f"Stage: {result['stage']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Probabilities:")
for stage, prob in result['probabilities'].items():
    print(f"  {stage}: {prob:.4f}")
```

**Output:**
```json
{
  "stage": "Stage II",
  "confidence": 0.9234,
  "probabilities": {
    "Stage I": 0.0512,
    "Stage II": 0.9234,
    "Stage III": 0.0198,
    "Stage IV": 0.0056
  },
  "enhanced_image": "base64_encoded_string..."
}
```

### Method 2: Batch Processing

```python
import cv2
from pathlib import Path

# Load images
images = [
    cv2.imread(f"path/to/image_{i}.jpg")
    for i in range(5)
]

# Batch predict
results = predictor.predict_batch(images, batch_size=8)

for i, result in enumerate(results):
    print(f"Image {i}: {result['stage']} ({result['confidence']:.4f})")
```

---

## 🌐 REST API

### Start the API Server

```bash
# Default: localhost:8000
python api/main.py

# Or with uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Server starts:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

### API Documentation

Interactive docs available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Endpoints

#### 1. Health Check
```bash
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "message": "Lung Cancer Detection API is running"
}
```

#### 2. Single Prediction

**Request:**
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -F "file=@ct_scan.jpg"
```

**Response:**
```json
{
  "stage": "Stage II",
  "confidence": 0.9234,
  "probabilities": {
    "Stage I": 0.0512,
    "Stage II": 0.9234,
    "Stage III": 0.0198,
    "Stage IV": 0.0056
  },
  "enhanced_image": "iVBORw0KGgoAAAANSUhEUgAAA..."
}
```

#### 3. Batch Prediction

**Request:**
```bash
curl -X POST "http://localhost:8000/api/predict-batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg" \
  -F "include_images=false"
```

**Response:**
```json
{
  "total_images": 3,
  "predictions": [
    {
      "stage": "Stage II",
      "confidence": 0.9234,
      "probabilities": {...}
    },
    {
      "stage": "Stage I",
      "confidence": 0.8756,
      "probabilities": {...}
    },
    {
      "stage": "Stage III",
      "confidence": 0.7892,
      "probabilities": {...}
    }
  ],
  "processing_time": 4.23
}
```

#### 4. Model Information

**Request:**
```bash
GET /api/model-info
```

**Response:**
```json
{
  "model_name": "efficientnet_b0",
  "num_classes": 4,
  "class_names": {
    "0": "Stage I",
    "1": "Stage II",
    "2": "Stage III",
    "3": "Stage IV"
  },
  "total_parameters": 4234321,
  "trainable_parameters": 523456,
  "device": "cuda",
  "checkpoint_epoch": 42
}
```

---

## 💻 Python Examples

### Example 1: Simple Prediction

```python
import sys
from pathlib import Path
sys.path.insert(0, "src")

from inference import create_predictor

# Load model
predictor = create_predictor(
    model_checkpoint="models/best_model.pth",
    device="cuda"
)

# Predict
result = predictor.predict_from_file("test_image.jpg")

print(f"Detected Stage: {result['stage']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Example 2: Batch Processing from Directory

```python
import cv2
from pathlib import Path
import sys
sys.path.insert(0, "src")

from inference import create_predictor

predictor = create_predictor("models/best_model.pth", device="cuda")

# Load all images from directory
image_dir = Path("test_images")
images = []
filenames = []

for img_path in sorted(image_dir.glob("*.jpg")):
    image = cv2.imread(str(img_path))
    images.append(image)
    filenames.append(img_path.name)

# Batch predict
results = predictor.predict_batch(images)

# Display results
for filename, result in zip(filenames, results):
    print(f"{filename}: {result['stage']} ({result['confidence']:.4f})")
```

### Example 3: Custom Inference with Preprocessing Visualization

```python
import cv2
import sys
sys.path.insert(0, "src")

from preprocessing import create_preprocessor
from inference import create_predictor

# Load image
image = cv2.imread("ct_scan.jpg")

# Preprocess separately to visualize
preprocessor = create_preprocessor(image_size=224)
tensor, enhanced = preprocessor.preprocess(image)

# Display original vs enhanced
cv2.imshow("Original", image)
cv2.imshow("Enhanced", enhanced)
cv2.waitKey(0)

# Predict
predictor = create_predictor("models/best_model.pth", device="cuda")
result = predictor.predict_from_array(image)

print(f"Stage: {result['stage']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## ⚙️ Configuration

Edit `src/config.py` to customize:

```python
# Model
MODEL_NAME = "efficientnet_b0"  # or resnet50
NUM_CLASSES = 4
IMAGE_SIZE = 224

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Preprocessing
CLAHE_CLIP_LIMIT = 2.0
FILTER_KERNEL_SIZE = 5

# API
API_HOST = "0.0.0.0"
API_PORT = 8000
```

---

## 📊 Model Details

### Architecture
- **Backbone**: EfficientNet-B0 (pretrained on ImageNet)
- **Head**: Custom fully-connected classifier
- **Layers**: Dropout + Dense(512) + ReLU + Dense(256) + ReLU + Dense(4)
- **Total Parameters**: ~4.2M

### Performance
- **Inference Time**: <2 seconds per image (GPU)
- **Memory**: ~1GB GPU
- **Input Size**: 224x224

### Image Preprocessing
1. Convert BGR → Grayscale
2. **CLAHE** (Contrast Limited Adaptive Histogram Equalization)
3. **Bilateral Filtering** (edge-preserving denoising)
4. Resize to 224×224
5. Normalize to [0, 1]
6. ImageNet normalization

---

## 🎯 Training Tips

### Best Practices
1. **Dataset**: Use 100+ images per class for best results
2. **Augmentation**: Already included (flip, rotation, brightness)
3. **Learning Rate**: Start with 1e-4, reduce if loss doesn't decrease
4. **Early Stopping**: Automatic after 10 epochs without improvement
5. **Model Selection**: EfficientNet-B0 balances speed and accuracy

### Hyperparameter Tuning
```bash
# Try different learning rates
python src/train.py --learning-rate 1e-3

# Adjust batch size
python src/train.py --batch-size 64

# Change number of epochs
python src/train.py --num-epochs 100
```

---

## 🧪 Testing

### Test Inference Speed

```python
import time
import cv2
import sys
sys.path.insert(0, "src")

from inference import create_predictor

predictor = create_predictor("models/best_model.pth", device="cuda")
test_image = cv2.imread("test.jpg")

# Warmup
predictor.predict_from_array(test_image)

# Benchmark
times = []
for _ in range(10):
    start = time.time()
    predictor.predict_from_array(test_image)
    times.append(time.time() - start)

avg_time = sum(times) / len(times)
print(f"Average inference time: {avg_time:.4f}s")
print(f"FPS: {1/avg_time:.2f}")
```

---

## 🔧 Troubleshooting

### GPU Not Being Used
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # GPU name
```

### Out of Memory
- Reduce `BATCH_SIZE` in config.py
- Use `device="cpu"` for inference (slower)
- Upgrade GPU or use smaller model

### Low Accuracy
- Check dataset balance (equal images per class)
- Increase training epochs
- Verify image quality and format
- Use data augmentation

---

## 📝 License

MIT License - Free for research and commercial use

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review API documentation at `/docs`
3. Check FastAPI documentation: https://fastapi.tiangolo.com/

---

## 🚀 Production Deployment

### Docker (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/
COPY models/ models/

CMD ["python", "api/main.py"]
```

Build and run:
```bash
docker build -t lung-cancer-api .
docker run -p 8000:8000 lung-cancer-api
```

### Production Checklist
- [ ] Use HTTPS for API
- [ ] Add authentication (JWT)
- [ ] Enable rate limiting
- [ ] Set up logging and monitoring
- [ ] Add request validation
- [ ] Use GPU with proper CUDA setup
- [ ] Set up health checks
- [ ] Implement caching

---

## 📊 Monitoring

Monitor API with built-in tools:
```bash
# Check API logs
tail -f logs/api.log

# Monitor GPU usage (if using NVIDIA)
nvidia-smi

# Check API health
curl http://localhost:8000/api/health
```

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Status**: Production Ready ✅
