"""
API Testing Script
Use this to test the REST API endpoints
"""

import json
import requests
from pathlib import Path
from typing import Dict, Any

BASE_URL = "http://localhost:8000"
API_PREFIX = "/api"


class APITester:
    """Test client for Lung Cancer Detection API"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Test health endpoint"""
        print("\n" + "="*60)
        print("TEST 1: Health Check")
        print("="*60)
        
        url = f"{self.base_url}{API_PREFIX}/health"
        print(f"GET {url}")
        
        response = self.session.get(url)
        print(f"Status: {response.status_code}")
        
        data = response.json()
        print(f"Response:")
        print(json.dumps(data, indent=2))
        
        return data
    
    def model_info(self) -> Dict[str, Any]:
        """Test model info endpoint"""
        print("\n" + "="*60)
        print("TEST 2: Model Information")
        print("="*60)
        
        url = f"{self.base_url}{API_PREFIX}/model-info"
        print(f"GET {url}")
        
        response = self.session.get(url)
        print(f"Status: {response.status_code}")
        
        data = response.json()
        print(f"Response:")
        print(json.dumps(data, indent=2))
        
        return data
    
    def predict(self, image_path: str, include_enhanced: bool = True) -> Dict[str, Any]:
        """Test single prediction endpoint"""
        print("\n" + "="*60)
        print("TEST 3: Single Prediction")
        print("="*60)
        
        image_path = Path(image_path)
        
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            return {}
        
        url = f"{self.base_url}{API_PREFIX}/predict"
        print(f"POST {url}")
        print(f"File: {image_path}")
        
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = self.session.post(url, files=files)
        
        print(f"Status: {response.status_code}")
        
        data = response.json()
        
        # Remove enhanced_image from display if too large
        if "enhanced_image" in data and not include_enhanced:
            data["enhanced_image"] = "[BASE64_IMAGE_DATA]"
        
        print(f"Response:")
        print(json.dumps(data, indent=2))
        
        return data
    
    def predict_batch(self, image_dir: str, include_images: bool = False) -> Dict[str, Any]:
        """Test batch prediction endpoint"""
        print("\n" + "="*60)
        print("TEST 4: Batch Prediction")
        print("="*60)
        
        image_dir = Path(image_dir)
        
        if not image_dir.exists():
            print(f"❌ Directory not found: {image_dir}")
            return {}
        
        # Find image files
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        
        if not image_files:
            print(f"❌ No images found in {image_dir}")
            return {}
        
        image_files = image_files[:5]  # Limit to 5 for testing
        
        url = f"{self.base_url}{API_PREFIX}/predict-batch"
        print(f"POST {url}")
        print(f"Images: {len(image_files)}")
        
        files = [("files", open(img, "rb")) for img in image_files]
        
        params = {"include_images": include_images}
        
        response = self.session.post(url, files=files, params=params)
        
        # Close files
        for _, f in files:
            f.close()
        
        print(f"Status: {response.status_code}")
        
        data = response.json()
        print(f"Response:")
        print(json.dumps(data, indent=2))
        
        return data


def main():
    """Run all API tests"""
    print("\n" + "█"*60)
    print("█ LUNG CANCER API - TEST SUITE")
    print("█"*60)
    print(f"\nTesting API at: {BASE_URL}")
    print("Make sure the API is running: python api/main.py")
    
    tester = APITester()
    
    try:
        # Test 1: Health check
        print("\n✅ Testing Health Check...")
        tester.health_check()
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API!")
        print("   Make sure the API is running: python api/main.py")
        return
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return
    
    try:
        # Test 2: Model info
        print("\n✅ Testing Model Info...")
        tester.model_info()
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
    
    # Example: Predict on single image
    print("\n✅ Testing Single Prediction...")
    print("   (Skipped - provide an image path)")
    print("   Usage: python -c \"from api_test import APITester; APITester().predict('path/to/image.jpg')\"")
    
    # Example: Batch prediction
    print("\n✅ Testing Batch Prediction...")
    print("   (Skipped - provide a directory path)")
    print("   Usage: python -c \"from api_test import APITester; APITester().predict_batch('path/to/images')\"")
    
    print("\n" + "█"*60)
    print("█ TESTS COMPLETED")
    print("█"*60)
    print("\nAPI Endpoints:")
    print("  GET  /api/health             - Health check")
    print("  GET  /api/model-info         - Model information")
    print("  POST /api/predict            - Single image prediction")
    print("  POST /api/predict-batch      - Batch prediction")
    print("\nAPI Documentation:")
    print("  Swagger UI: http://localhost:8000/docs")
    print("  ReDoc:      http://localhost:8000/redoc")
    print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "health":
            APITester().health_check()
        elif command == "info":
            APITester().model_info()
        elif command == "predict" and len(sys.argv) > 2:
            APITester().predict(sys.argv[2])
        elif command == "batch" and len(sys.argv) > 2:
            APITester().predict_batch(sys.argv[2])
        else:
            print("Usage:")
            print("  python api_test.py               # Run all tests")
            print("  python api_test.py health        # Test health")
            print("  python api_test.py info          # Get model info")
            print("  python api_test.py predict <img> # Single prediction")
            print("  python api_test.py batch <dir>   # Batch prediction")
    else:
        main()
