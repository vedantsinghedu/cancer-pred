"""
Standalone training script for the lung cancer detection model.
Run this script to train or fine-tune the model on your dataset.

Usage:
    python train_standalone.py --data-dir data --batch-size 32 --num-epochs 50
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import create_model
from train import Trainer
from preprocessing import ImagePreprocessor
from utils import setup_logger, create_data_loaders

logger = setup_logger(__name__, log_file="models/training_log.txt")


def create_dataset(
    data_dir: str,
    image_size: int = 224,
    augment: bool = True,
) -> datasets.ImageFolder:
    """
    Create dataset from directory structure.
    
    Expected structure:
        data/
        ├── stage_1/
        │   ├── image1.jpg
        │   └── ...
        ├── stage_2/
        └── ...
    
    Args:
        data_dir: Root data directory
        image_size: Target image size
        augment: Apply data augmentation
        
    Returns:
        PyTorch ImageFolder dataset
    """
    # Define transforms
    if augment:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    
    # Create dataset
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    logger.info(f"Dataset loaded from {data_dir}")
    logger.info(f"Total images: {len(dataset)}")
    logger.info(f"Classes: {dataset.classes}")
    logger.info(f"Class to idx: {dataset.class_to_idx}")
    
    return dataset


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(
        description="Train Lung Cancer Detection Model"
    )
    
    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root data directory with class subdirectories",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/",
        help="Directory to save checkpoints",
    )
    
    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="efficientnet_b0",
        choices=["efficientnet_b0", "efficientnet_b1", "resnet50", "resnet101"],
        help="Model architecture",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=4,
        help="Number of classification classes",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Target image size",
    )
    
    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for optimizer",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience in epochs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for training",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1,
        help="Save checkpoint every N epochs",
    )
    
    args = parser.parse_args()
    
    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    logger.info("=" * 60)
    logger.info("LUNG CANCER DETECTION - TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Checkpoint directory: {args.checkpoint_dir}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Device: {args.device}")
    
    # Create dataset
    logger.info("\nCreating dataset...")
    dataset = create_dataset(
        str(data_dir),
        image_size=args.image_size,
        augment=True,
    )
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    logger.info("\nCreating model...")
    model = create_model(
        model_name=args.model_name,
        num_classes=args.num_classes,
        pretrained=True,
    )
    
    logger.info(f"Total parameters: {model.get_total_params():,}")
    logger.info(f"Trainable parameters: {model.get_trainable_params():,}")
    
    # Create trainer
    logger.info("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience,
    )
    
    # Train
    logger.info("\nStarting training loop...")
    logger.info("=" * 60)
    
    history = trainer.train(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
    )
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Best train loss: {min(history['train_loss']):.4f}")
    logger.info(f"Best val loss: {min(history['val_loss']):.4f}")
    logger.info(f"Best train accuracy: {max(history['train_acc']):.4f}")
    logger.info(f"Best val accuracy: {max(history['val_acc']):.4f}")
    logger.info(f"Checkpoints saved to: {args.checkpoint_dir}")
    logger.info("=" * 60)
    
    print("\n✅ Training completed!")
    print(f"   Best model saved to: {Path(args.checkpoint_dir) / 'best_model.pth'}")
    print(f"\nNext steps:")
    print(f"   1. Test the model: python examples.py")
    print(f"   2. Start API: python api/main.py")


if __name__ == "__main__":
    main()
