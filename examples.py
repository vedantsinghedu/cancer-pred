"""
Example usage script demonstrating the complete pipeline.
Shows single prediction, batch prediction, and model info retrieval.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
from inference import create_predictor
from utils import setup_logger

logger = setup_logger(__name__)


def example_single_prediction():
    """Example: Single image prediction"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Image Prediction")
    print("="*60)
    
    # Initialize predictor
    model_path = Path(__file__).parent / "models" / "best_model.pth"
    
    if not model_path.exists():
        print("⚠️  Model not found at:", model_path)
        print("    Please train the model first using: python src/train.py")
        return
    
    print("Loading model...")
    predictor = create_predictor(
        model_checkpoint=str(model_path),
        model_name="efficientnet_b0",
        num_classes=4,
        device="cuda",
    )
    
    print("\n✅ Model loaded successfully!")
    
    # Get model info
    info = predictor.get_model_info()
    print("\nModel Information:")
    print(f"  Architecture: {info['model_name']}")
    print(f"  Classes: {info['num_classes']}")
    print(f"  Total Parameters: {info['total_parameters']:,}")
    print(f"  Trainable Parameters: {info['trainable_parameters']:,}")
    print(f"  Device: {info['device']}")
    
    # Create dummy image for demonstration
    print("\nCreating dummy CT scan image for demonstration...")
    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Predict
    print("Running inference...")
    result = predictor.predict_from_array(dummy_image)
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Predicted Stage: {result['stage']}")
    print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
    print("\nClass Probabilities:")
    for stage, prob in result['probabilities'].items():
        bar = "█" * int(prob * 50)
        print(f"  {stage:12s}: {prob:.4f} {bar}")


def example_model_info():
    """Example: Get model information"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Model Information")
    print("="*60)
    
    model_path = Path(__file__).parent / "models" / "best_model.pth"
    
    if not model_path.exists():
        print("⚠️  Model not found at:", model_path)
        return
    
    print("Loading model...")
    predictor = create_predictor(
        model_checkpoint=str(model_path),
        model_name="efficientnet_b0",
        num_classes=4,
        device="cuda",
    )
    
    # Get detailed info
    info = predictor.get_model_info()
    
    print("\n" + "="*60)
    print("MODEL DETAILS")
    print("="*60)
    print(f"Architecture:          {info['model_name']}")
    print(f"Number of Classes:     {info['num_classes']}")
    print(f"Device:                {info['device']}")
    print(f"Total Parameters:      {info['total_parameters']:,}")
    print(f"Trainable Parameters:  {info['trainable_parameters']:,}")
    print(f"Checkpoint Epoch:      {info['checkpoint_epoch']}")
    
    print("\nClass Mappings:")
    for idx, name in info['class_names'].items():
        print(f"  Class {idx}: {name}")


def example_batch_prediction():
    """Example: Batch prediction on multiple images"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch Prediction")
    print("="*60)
    
    model_path = Path(__file__).parent / "models" / "best_model.pth"
    
    if not model_path.exists():
        print("⚠️  Model not found at:", model_path)
        return
    
    print("Loading model...")
    predictor = create_predictor(
        model_checkpoint=str(model_path),
        model_name="efficientnet_b0",
        num_classes=4,
        device="cuda",
    )
    
    # Create dummy images
    print("\nCreating 5 dummy CT scan images...")
    images = [
        np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        for _ in range(5)
    ]
    
    # Batch predict
    print("Running batch inference...")
    results = predictor.predict_batch(images, batch_size=2)
    
    # Display results
    print("\n" + "="*60)
    print("BATCH PREDICTION RESULTS")
    print("="*60)
    
    for i, result in enumerate(results, 1):
        print(f"\nImage {i}:")
        print(f"  Stage: {result['stage']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Probabilities:")
        for stage, prob in result['probabilities'].items():
            print(f"    {stage}: {prob:.4f}")


def main():
    """Run all examples"""
    print("\n" + "█"*60)
    print("█ LUNG CANCER DETECTION SYSTEM - USAGE EXAMPLES")
    print("█"*60)
    
    print("\nNote: These examples use dummy images for demonstration.")
    print("For real predictions, use your actual CT scan images.")
    
    try:
        example_single_prediction()
    except Exception as e:
        print(f"❌ Example 1 failed: {str(e)}")
    
    try:
        example_model_info()
    except Exception as e:
        print(f"❌ Example 2 failed: {str(e)}")
    
    try:
        example_batch_prediction()
    except Exception as e:
        print(f"❌ Example 3 failed: {str(e)}")
    
    print("\n" + "█"*60)
    print("█ EXAMPLES COMPLETED")
    print("█"*60)
    print("\nNext Steps:")
    print("  1. Train the model: python src/train.py")
    print("  2. Start the API:   python api/main.py")
    print("  3. Make predictions: curl -X POST http://localhost:8000/api/predict ...")
    print()


if __name__ == "__main__":
    main()
