# 🧪 TESTING & VALIDATION GUIDE

## Unit Tests for Core Components

### Test 1: Image Preprocessing
```python
import sys
sys.path.insert(0, 'src')

import numpy as np
import cv2
from preprocessing import create_preprocessor

# Create test image
test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

# Initialize preprocessor
preprocessor = create_preprocessor(image_size=224)

# Test preprocessing
tensor, enhanced = preprocessor.preprocess(test_image)

# Validate output
assert tensor.shape == (3, 224, 224), f"Wrong tensor shape: {tensor.shape}"
assert enhanced.shape == (224, 224, 3), f"Wrong image shape: {enhanced.shape}"
assert tensor.min() >= -2 and tensor.max() <= 2, "Tensor values out of range"
assert enhanced.min() >= 0 and enhanced.max() <= 255, "Enhanced image out of range"

print("✅ Preprocessing tests passed!")
```

### Test 2: Model Loading
```python
import torch
from model import create_model, load_model

# Test model creation
model = create_model(
    model_name="efficientnet_b0",
    num_classes=4,
    pretrained=True
)

# Validate
assert model.get_total_params() > 0, "Model has no parameters"
print(f"✅ Model created: {model.get_total_params():,} total params")

# Test inference shape
test_input = torch.randn(2, 3, 224, 224)
output = model(test_input)
assert output.shape == (2, 4), f"Wrong output shape: {output.shape}"
print("✅ Model inference shape test passed!")
```

### Test 3: Inference Pipeline
```python
import numpy as np
import cv2
from inference import create_predictor

# Create dummy image
dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

# Create predictor (requires trained model)
try:
    predictor = create_predictor(
        model_checkpoint="models/best_model.pth",
        device="cuda"
    )
    
    # Single prediction
    result = predictor.predict_from_array(dummy_image)
    
    # Validate result
    assert "stage" in result, "Missing 'stage' in result"
    assert "confidence" in result, "Missing 'confidence' in result"
    assert "probabilities" in result, "Missing 'probabilities' in result"
    assert 0 <= result["confidence"] <= 1, "Confidence out of range"
    
    print("✅ Inference pipeline test passed!")
    
except FileNotFoundError:
    print("⚠️  Model not found - train model first")
```

---

## API Integration Tests

### Test 1: Health Check
```bash
curl -X GET http://localhost:8000/api/health

# Expected response:
# {"status":"healthy","version":"1.0.0","message":"..."}
```

### Test 2: Model Info
```bash
curl -X GET http://localhost:8000/api/model-info

# Expected response:
# {
#   "model_name": "efficientnet_b0",
#   "num_classes": 4,
#   "total_parameters": 4234321,
#   ...
# }
```

### Test 3: Single Prediction
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -F "file=@path/to/ct_scan.jpg" \
  -H "Accept: application/json"

# Expected response:
# {
#   "stage": "Stage II",
#   "confidence": 0.9234,
#   "probabilities": {...},
#   "enhanced_image": "base64_string..."
# }
```

### Test 4: Error Handling
```bash
# Test invalid file format
curl -X POST "http://localhost:8000/api/predict" \
  -F "file=@document.pdf"

# Expected: 400 Bad Request
# {"detail": "Invalid file format..."}

# Test missing file
curl -X POST "http://localhost:8000/api/predict"

# Expected: 400 Bad Request
# {"detail": "No file provided"}
```

---

## Performance Benchmarking

### Benchmark 1: Preprocessing Speed
```python
import time
import numpy as np
from preprocessing import create_preprocessor

preprocessor = create_preprocessor()
test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

# Warmup
preprocessor.preprocess(test_image)

# Benchmark
times = []
for _ in range(100):
    start = time.time()
    preprocessor.preprocess(test_image)
    times.append(time.time() - start)

avg_time = sum(times) / len(times)
std_time = (sum((t - avg_time)**2 for t in times) / len(times))**0.5

print(f"Preprocessing Speed:")
print(f"  Average: {avg_time*1000:.2f}ms")
print(f"  Std Dev: {std_time*1000:.2f}ms")
print(f"  Min:     {min(times)*1000:.2f}ms")
print(f"  Max:     {max(times)*1000:.2f}ms")
```

### Benchmark 2: Inference Speed
```python
import time
import torch
import numpy as np
from inference import create_predictor

predictor = create_predictor("models/best_model.pth", device="cuda")
test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

# Warmup
predictor.predict_from_array(test_image)

# Benchmark
times = []
for _ in range(50):
    start = time.time()
    predictor.predict_from_array(test_image)
    times.append(time.time() - start)

avg_time = sum(times) / len(times)
fps = 1 / avg_time

print(f"Inference Speed:")
print(f"  Average: {avg_time*1000:.2f}ms")
print(f"  FPS:     {fps:.2f}")
print(f"  Total:   {2*avg_time*1000:.2f}ms (with preprocessing)")
```

### Benchmark 3: Batch Processing
```python
import time
import numpy as np
from inference import create_predictor

predictor = create_predictor("models/best_model.pth", device="cuda")

# Create batch of images
batch_sizes = [1, 4, 8, 16, 32]
results = {}

for batch_size in batch_sizes:
    images = [
        np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        for _ in range(batch_size)
    ]
    
    # Warmup
    predictor.predict_batch(images, batch_size=batch_size)
    
    # Benchmark
    start = time.time()
    for _ in range(10):
        predictor.predict_batch(images, batch_size=batch_size)
    total_time = (time.time() - start) / 10
    
    per_image = total_time / batch_size
    results[batch_size] = {
        "total": total_time,
        "per_image": per_image,
        "fps": 1 / per_image
    }

print("Batch Processing Benchmark:")
for batch_size, metrics in results.items():
    print(f"  Batch {batch_size:2d}: {metrics['total']*1000:6.2f}ms total, "
          f"{metrics['per_image']*1000:5.2f}ms/img, {metrics['fps']:5.1f} FPS")
```

---

## Accuracy Validation

### Validation Script
```python
import sys
sys.path.insert(0, 'src')

import torch
from pathlib import Path
from inference import create_predictor
from utils import get_class_name

predictor = create_predictor("models/best_model.pth", device="cuda")

# Test on known images
test_images_dir = Path("test_images")

# Expected results
expected_results = {
    "image1.jpg": "Stage I",
    "image2.jpg": "Stage II",
    # Add more test cases
}

correct = 0
total = 0

for img_file, expected_stage in expected_results.items():
    img_path = test_images_dir / img_file
    
    if img_path.exists():
        result = predictor.predict_from_file(str(img_path))
        predicted_stage = result["stage"]
        confidence = result["confidence"]
        
        is_correct = predicted_stage == expected_stage
        correct += int(is_correct)
        total += 1
        
        status = "✅" if is_correct else "❌"
        print(f"{status} {img_file}: Predicted {predicted_stage} "
              f"({confidence:.2%}), Expected {expected_stage}")

accuracy = correct / total if total > 0 else 0
print(f"\nAccuracy: {accuracy:.1%} ({correct}/{total})")
```

---

## Memory Profiling

### GPU Memory Usage
```python
import torch
import numpy as np
from inference import create_predictor

# Check initial GPU memory
print(f"GPU Memory:")
print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"  Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Load predictor
predictor = create_predictor("models/best_model.pth", device="cuda")

print(f"After loading model:")
print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"  Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Make predictions
images = [np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8) for _ in range(8)]
results = predictor.predict_batch(images)

print(f"After batch prediction (8 images):")
print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"  Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

---

## Load Testing

### API Load Test (using locust)
```python
# Install: pip install locust

from locust import HttpUser, task, between
import tempfile
import numpy as np
import cv2

class APILoadTest(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        # Create test image
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', img)
        self.test_image = buffer.tobytes()
    
    @task(1)
    def health_check(self):
        self.client.get("/api/health")
    
    @task(3)
    def predict(self):
        files = {'file': ('test.jpg', self.test_image)}
        self.client.post("/api/predict", files=files)
    
    @task(1)
    def model_info(self):
        self.client.get("/api/model-info")

# Run: locust -f load_test.py --host=http://localhost:8000
```

---

## Continuous Integration Checklist

- [ ] Code formatting (black, flake8)
- [ ] Type checking (mypy)
- [ ] Unit tests (pytest)
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] Code coverage >80%
- [ ] Security scanning (bandit)
- [ ] Dependency vulnerabilities (safety)

---

## Pre-Deployment Checklist

- [ ] Train model on complete dataset
- [ ] Validate accuracy on test set
- [ ] Run performance benchmarks
- [ ] Load test with expected traffic
- [ ] Test error handling
- [ ] Verify reproducibility
- [ ] Check GPU memory requirements
- [ ] Test on target hardware
- [ ] Security audit
- [ ] Documentation review

---

## Monitoring in Production

### Key Metrics
1. **Inference metrics**
   - Response time (p50, p95, p99)
   - Throughput (requests/second)
   - Error rate

2. **Model metrics**
   - Prediction distribution
   - Confidence distribution
   - Class balance

3. **System metrics**
   - GPU utilization
   - GPU memory
   - CPU usage
   - Request queue length

### Logging
```python
# Example: Log predictions
import logging
from datetime import datetime

logger = logging.getLogger("prediction")

def log_prediction(image_path, result):
    logger.info(
        f"Prediction|{datetime.now()}|"
        f"image={image_path}|"
        f"stage={result['stage']}|"
        f"confidence={result['confidence']:.4f}"
    )
```

---

## Troubleshooting Test Failures

| Issue | Cause | Solution |
|-------|-------|----------|
| Out of Memory | Batch too large | Reduce batch size |
| Slow inference | CPU used | Enable GPU |
| Model not found | Wrong path | Check path in config |
| API not responding | Port in use | Change port |
| Low accuracy | Model undertrained | Train more epochs |
| Preprocessing corrupts image | Incorrect normalization | Verify preprocessing |

---

**Testing Guide Version**: 1.0  
**Last Updated**: 2024  
**Status**: Production Ready ✅
