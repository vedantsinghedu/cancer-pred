"""
Synthetic Test Dataset Generator
Creates realistic synthetic CT scan images for testing the training pipeline.
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw


def create_synthetic_ct_image(stage: int, size: int = 256) -> np.ndarray:
    """
    Create a realistic synthetic CT scan image.
    
    Args:
        stage: Cancer stage (1-4)
        size: Image size (256x256)
        
    Returns:
        Synthetic CT image as numpy array
    """
    # Start with noise background (CT-like texture)
    image = np.random.randint(40, 120, (size, size), dtype=np.uint8)
    
    # Add some gaussian smoothing to make it look more like CT
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Add lungs (darker areas)
    center_x, center_y = size // 2, size // 2
    
    # Left lung
    cv2.ellipse(
        image,
        (center_x - size // 4, center_y),
        (size // 6, size // 4),
        0,
        0,
        360,
        (80, 80, 80),
        -1,
    )
    
    # Right lung
    cv2.ellipse(
        image,
        (center_x + size // 4, center_y),
        (size // 6, size // 4),
        0,
        0,
        360,
        (80, 80, 80),
        -1,
    )
    
    # Add nodules/tumors based on stage
    num_nodules = stage  # Stage 1 = 1 nodule, Stage 4 = 4 nodules
    nodule_sizes = [10 + stage * 5, 12 + stage * 5]  # Larger for worse stages
    
    for i in range(num_nodules):
        # Random position in lungs
        angle = (360 // num_nodules) * i + np.random.randint(-30, 30)
        radius_from_center = size // 5 + np.random.randint(-20, 20)
        
        x = int(center_x + radius_from_center * np.cos(np.radians(angle)))
        y = int(center_y + radius_from_center * np.sin(np.radians(angle)))
        
        # Clamp to image
        x = max(30, min(size - 30, x))
        y = max(30, min(size - 30, y))
        
        # Draw nodule (brighter = more severe)
        nodule_size = nodule_sizes[0] + i * 3
        brightness = 150 + stage * 20  # Brighter for worse stages
        
        cv2.circle(image, (x, y), nodule_size, brightness, -1)
        
        # Add some texture to nodule
        for _ in range(5):
            offset_x = np.random.randint(-nodule_size // 2, nodule_size // 2)
            offset_y = np.random.randint(-nodule_size // 2, nodule_size // 2)
            cv2.circle(
                image,
                (x + offset_x, y + offset_y),
                2,
                brightness + np.random.randint(-20, 20),
                -1,
            )
    
    # Add some noise for realism
    noise = np.random.normal(0, 5, (size, size))
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    # Convert to 3-channel BGR
    bgr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    return bgr_image


def create_dataset(
    output_dir: str = "data",
    num_per_stage: int = 30,
    image_size: int = 256,
    verbose: bool = True,
) -> dict:
    """
    Create synthetic dataset with proper directory structure.
    
    Args:
        output_dir: Output directory root
        num_per_stage: Number of images per stage
        image_size: Image size
        verbose: Print progress
        
    Returns:
        Dictionary with creation statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    stats = {"total": 0, "by_stage": {}}
    
    for stage in range(1, 5):
        stage_dir = output_path / f"stage_{stage}"
        stage_dir.mkdir(exist_ok=True, parents=True)
        
        if verbose:
            print(f"\n📁 Creating Stage {stage} images...")
        
        for i in range(num_per_stage):
            # Create synthetic image
            image = create_synthetic_ct_image(stage, size=image_size)
            
            # Save image
            filename = f"synthetic_stage{stage}_{i:04d}.jpg"
            filepath = stage_dir / filename
            cv2.imwrite(str(filepath), image)
            
            stats["total"] += 1
            stats["by_stage"][stage] = stats["by_stage"].get(stage, 0) + 1
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  ✓ Created {i + 1}/{num_per_stage} images")
        
        if verbose:
            print(f"  ✅ Stage {stage}: {num_per_stage} images created")
    
    return stats


def display_sample_images(output_dir: str = "data", num_samples: int = 1):
    """
    Display sample images from each stage.
    
    Args:
        output_dir: Dataset directory
        num_samples: Number of samples to display per stage
    """
    print("\n📊 Dataset Summary:")
    print("=" * 60)
    
    output_path = Path(output_dir)
    total_images = 0
    
    for stage in range(1, 5):
        stage_dir = output_path / f"stage_{stage}"
        
        if stage_dir.exists():
            images = list(stage_dir.glob("*.jpg"))
            count = len(images)
            total_images += count
            
            print(f"Stage {stage}: {count} images")
            if count > 0 and count >= num_samples:
                print(f"  Sample: {images[0].name}")
    
    print("=" * 60)
    print(f"Total images: {total_images}")
    print(f"Directory: {output_path.resolve()}")


def validate_dataset(output_dir: str = "data") -> bool:
    """
    Validate dataset structure and files.
    
    Args:
        output_dir: Dataset directory to validate
        
    Returns:
        True if valid, False otherwise
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"❌ Dataset directory not found: {output_dir}")
        return False
    
    valid = True
    
    for stage in range(1, 5):
        stage_dir = output_path / f"stage_{stage}"
        
        if not stage_dir.exists():
            print(f"❌ Stage {stage} directory not found")
            valid = False
            continue
        
        images = list(stage_dir.glob("*.jpg"))
        if len(images) == 0:
            print(f"❌ Stage {stage} has no images")
            valid = False
        else:
            print(f"✅ Stage {stage}: {len(images)} images")
            
            # Check image validity
            try:
                img = cv2.imread(str(images[0]))
                if img is None:
                    print(f"   ⚠️ Could not read sample image")
                else:
                    print(f"   ✓ Image shape: {img.shape}")
            except Exception as e:
                print(f"   ⚠️ Error reading image: {str(e)}")
    
    return valid


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate synthetic CT scan dataset for testing"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--num-per-stage",
        type=int,
        default=30,
        help="Number of images per stage",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Image size (height and width)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing dataset",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🧪 SYNTHETIC CT DATASET GENERATOR")
    print("=" * 60)
    
    if args.validate_only:
        print("\n🔍 Validating dataset...")
        if validate_dataset(args.output_dir):
            print("\n✅ Dataset is valid!")
            return 0
        else:
            print("\n❌ Dataset validation failed!")
            return 1
    
    print(f"\n📋 Configuration:")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Images per stage: {args.num_per_stage}")
    print(f"  Image size: {args.image_size}×{args.image_size}")
    print(f"  Total images: {args.num_per_stage * 4}")
    
    # Create dataset
    print("\n🎨 Creating synthetic dataset...")
    stats = create_dataset(
        output_dir=args.output_dir,
        num_per_stage=args.num_per_stage,
        image_size=args.image_size,
        verbose=not args.quiet,
    )
    
    print("\n✅ Dataset creation complete!")
    
    # Display summary
    display_sample_images(args.output_dir)
    
    # Validate
    print("\n🔍 Validating dataset...")
    if validate_dataset(args.output_dir):
        print("\n✅ Dataset is ready for training!")
        print("\n🚀 Next steps:")
        print(f"   1. Verify dataset: python verify_setup.py")
        print(f"   2. Train model:   python train_standalone.py --data-dir {args.output_dir}")
        print(f"   3. Start API:     python api/main.py")
        return 0
    else:
        print("\n❌ Dataset validation failed!")
        return 1


if __name__ == "__main__":
    exit(main())
