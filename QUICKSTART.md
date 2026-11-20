# 🚀 QUICK START GUIDE

## Installation (5 minutes)

### 1. Clone and Setup
```bash
cd lung-cancer-detection
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Training (First Time Setup)

### 1. Prepare Dataset
Organize images in subdirectories by class:
```
data/
├── stage_1/
│   ├── image1.jpg
│   └── ...
├── stage_2/
├── stage_3/
└── stage_4/
```

### 2. Train Model
```bash
# Simple training
python train_standalone.py --data-dir data

# With custom parameters
python train_standalone.py --data-dir data \
    --batch-size 32 \
    --num-epochs 50 \
    --learning-rate 1e-4 \
    --model-name efficientnet_b0
```

**Output:**
- `models/best_model.pth` (best checkpoint)
- `models/training_log.txt` (logs)

---

## Using the System

### Option A: Python Code
```python
import sys
sys.path.insert(0, 'src')

from inference import create_predictor
import cv2

# Load model
predictor = create_predictor(
    model_checkpoint="models/best_model.pth",
    device="cuda"
)

# Predict
image = cv2.imread("ct_scan.jpg")
result = predictor.predict_from_array(image)

print(f"Stage: {result['stage']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Option B: REST API
```bash
# Start server
python api/main.py

# In another terminal, make predictions
curl -X POST "http://localhost:8000/api/predict" \
  -F "file=@ct_scan.jpg"
```

### Option C: Web UI
Open browser: `http://localhost:8000/docs`

---

## Common Commands

```bash
# Train model
python train_standalone.py --data-dir data

# Start API
python api/main.py

# Test API (requires running server)
python api_test.py

# Run examples
python examples.py

# Health check
curl http://localhost:8000/api/health

# Single prediction
curl -X POST "http://localhost:8000/api/predict" -F "file=@image.jpg"

# Get model info
curl http://localhost:8000/api/model-info
```

---

## File Descriptions

| File | Purpose |
|------|---------|
| `src/config.py` | Settings and configuration |
| `src/preprocessing.py` | Image enhancement (CLAHE, filtering) |
| `src/model.py` | Model architecture |
| `src/inference.py` | Prediction pipeline |
| `src/train.py` | Training logic |
| `api/main.py` | FastAPI server |
| `api/routes.py` | API endpoints |
| `api/schemas.py` | Data validation |
| `train_standalone.py` | Standalone training script |
| `examples.py` | Usage examples |
| `api_test.py` | API testing script |

---

## Troubleshooting

### GPU Not Working
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))
```

If False, install CPU version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Out of Memory
- Reduce `BATCH_SIZE` in `src/config.py`
- Use CPU: `device="cpu"` (slower)

### Model Not Found
```bash
# Train a model first
python train_standalone.py --data-dir data
```

### Port Already in Use
```bash
# Use different port
python -c "import uvicorn; uvicorn.run('api.main:app', port=8001)"
```

---

## API Examples

### 1. Health Check
```bash
curl http://localhost:8000/api/health
```

### 2. Get Model Info
```bash
curl http://localhost:8000/api/model-info
```

### 3. Predict Single Image
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -F "file=@lung_scan.jpg"
```

### 4. Batch Prediction
```bash
curl -X POST "http://localhost:8000/api/predict-batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

### 5. With Python Requests
```python
import requests

# Single image
files = {'file': open('ct_scan.jpg', 'rb')}
response = requests.post('http://localhost:8000/api/predict', files=files)
result = response.json()
print(f"Stage: {result['stage']}, Confidence: {result['confidence']:.2%}")
```

---

## Performance Tips

### Faster Training
- Use GPU (50x faster than CPU)
- Increase batch size (if memory allows)
- Use fewer epochs initially

### Faster Inference
- Use batch processing for multiple images
- Use FP16 (half precision) on GPU
- Use smaller model: EfficientNet-B0 < B1 < B2

### Better Accuracy
- Use more training data (100+ per class)
- Fine-tune more epochs
- Use data augmentation
- Reduce learning rate if overfitting

---

## Next Steps

1. ✅ Install dependencies
2. ✅ Prepare dataset
3. ✅ Train model
4. ✅ Start API
5. ✅ Make predictions
6. 📈 Monitor performance
7. 🚀 Deploy to production

---

## Need Help?

- Check `README.md` for detailed documentation
- Run `examples.py` for code examples
- Run `api_test.py` to test API endpoints
- Check logs in `models/training_log.txt`

---

**Status**: ✅ Ready for production use

**Version**: 1.0.0
