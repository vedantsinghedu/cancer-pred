"""
Training module for fine-tuning the lung cancer detection model.
Implements training loop with early stopping and model checkpointing.
"""

import time
from pathlib import Path
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from model import LungCancerModel, create_model, save_model
from utils import setup_logger, get_device

logger = setup_logger(__name__)


class EarlyStoppingCallback:
    """Early stopping callback to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_epoch = 0
    
    def __call__(self, val_loss: float, epoch: int) -> bool:
        """
        Check if early stopping criterion is met.
        
        Args:
            val_loss: Validation loss
            epoch: Current epoch
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(
                    f"Early stopping triggered. Best epoch: {self.best_epoch}, "
                    f"Best loss: {self.best_loss:.4f}"
                )
                return True
        
        return False


class Trainer:
    """
    Trainer for lung cancer detection model.
    Handles training loop, validation, and model checkpointing.
    """
    
    def __init__(
        self,
        model: LungCancerModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            num_epochs: Number of epochs
            early_stopping_patience: Early stopping patience
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = get_device() if device == "cuda" else torch.device("cpu")
        self.num_epochs = num_epochs
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=True,
        )
        
        # Setup early stopping
        self.early_stopping = EarlyStoppingCallback(
            patience=early_stopping_patience
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }
        
        logger.info(f"Trainer initialized. Total params: {model.get_total_params():,}")
        logger.info(f"Trainable params: {model.get_trainable_params():,}")
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"  Batch [{batch_idx + 1}/{len(self.train_loader)}], "
                    f"Loss: {loss.item():.4f}"
                )
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def train(
        self,
        checkpoint_dir: Path,
        checkpoint_interval: int = 1,
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            checkpoint_interval: Save checkpoint every N epochs
            
        Returns:
            Training history dictionary
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        logger.info(f"Starting training for {self.num_epochs} epochs...")
        logger.info(f"Device: {self.device}")
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Log
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            epoch_time = time.time() - epoch_start
            
            logger.info(
                f"Epoch [{epoch + 1}/{self.num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Save checkpoint
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1}.pth"
                save_model(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    loss=val_loss,
                    checkpoint_path=str(checkpoint_path),
                )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = checkpoint_dir / "best_model.pth"
                save_model(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    loss=val_loss,
                    checkpoint_path=str(best_checkpoint_path),
                    is_best=True,
                )
            
            # Early stopping
            if self.early_stopping(val_loss, epoch + 1):
                break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        
        return self.history


def train_pipeline(
    dataset,
    batch_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    model_name: str = "efficientnet_b0",
    num_classes: int = 4,
    checkpoint_dir: str = "models/",
    device: str = "cuda",
) -> Tuple[LungCancerModel, Dict]:
    """
    Complete training pipeline.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        num_epochs: Number of epochs
        learning_rate: Learning rate
        model_name: Model architecture
        num_classes: Number of classes
        checkpoint_dir: Directory to save checkpoints
        device: Device to use
        
    Returns:
        Tuple of (trained model, training history)
    """
    from utils import create_data_loaders
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset,
        batch_size=batch_size,
    )
    
    logger.info(
        f"Data loaders created. "
        f"Train: {len(train_loader)}, Val: {len(val_loader)}, "
        f"Test: {len(test_loader)}"
    )
    
    # Create model
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=True,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )
    
    # Train
    history = trainer.train(
        checkpoint_dir=checkpoint_dir,
    )
    
    return model, history
