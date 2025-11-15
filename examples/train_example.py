#!/usr/bin/env python3
"""
TFAN Training Example
End-to-end training pipeline for deployment.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tfan.model import TFANModel
from tfan.trainer import TFANTrainer, FDTScheduler
from tfan.datasets import create_dataloaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_cifar10(
    epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "./checkpoints",
    use_fdt: bool = True
):
    """
    Train TFAN model on CIFAR-10.

    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        device: Device to train on
        output_dir: Directory to save checkpoints
        use_fdt: Use FDT homeostatic scheduler
    """
    logger.info("=" * 80)
    logger.info("TFAN Training on CIFAR-10")
    logger.info("=" * 80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # 1. Load Data
    # ========================================================================
    logger.info("Loading CIFAR-10 dataset...")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    # ========================================================================
    # 2. Create Model
    # ========================================================================
    logger.info("Creating TFAN model...")

    # For CIFAR-10: 32x32 RGB images → 10 classes
    # We'll treat images as sequences for TFAN
    model_config = {
        'input_dim': 32 * 32 * 3,  # Flattened image
        'hidden_dim': 256,
        'output_dim': 10,
        'num_heads': 8,
        'num_layers': 4,
        'seq_len': 1,  # Single timestep (image classification)
        'dropout': 0.1,
        'use_topology': True,
        'use_sparse_attention': True,
        'use_hyperbolic': True
    }

    model = TFANModel(**model_config)
    model = model.to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ========================================================================
    # 3. Setup Training
    # ========================================================================
    logger.info("Setting up training...")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )

    if use_fdt:
        # Use FDT homeostatic scheduler
        scheduler = FDTScheduler(
            optimizer,
            initial_lr=learning_rate,
            initial_temp=1.0,
            target_epr_cv=0.15,
            pid_kp=0.30,
            pid_ki=0.02,
            pid_kd=0.10,
            adaptive_gains=True
        )
    else:
        # Standard cosine annealing
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )

    # ========================================================================
    # 4. Training Loop
    # ========================================================================
    logger.info("Starting training...")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"FDT scheduler: {use_fdt}")

    best_accuracy = 0.0
    train_history = {
        'train_loss': [],
        'test_loss': [],
        'test_accuracy': [],
        'learning_rate': []
    }

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Reshape for TFAN: [B, C, H, W] → [B, 1, C*H*W]
            inputs = inputs.view(inputs.size(0), 1, -1)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Get logits (assuming model outputs [B, 1, num_classes])
            if outputs.dim() == 3:
                outputs = outputs.squeeze(1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 100 == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs} [{batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Acc: {100.0*train_correct/train_total:.2f}%"
                )

        # Evaluate
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.view(inputs.size(0), 1, -1)

                outputs = model(inputs)
                if outputs.dim() == 3:
                    outputs = outputs.squeeze(1)

                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        # Statistics
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = 100.0 * test_correct / test_total

        train_history['train_loss'].append(avg_train_loss)
        train_history['test_loss'].append(avg_test_loss)
        train_history['test_accuracy'].append(test_accuracy)
        train_history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        logger.info(
            f"Epoch {epoch}/{epochs} - "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Test Loss: {avg_test_loss:.4f} "
            f"Test Acc: {test_accuracy:.2f}% "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Update scheduler
        if use_fdt:
            # FDT needs loss values
            epr_value = avg_train_loss / (avg_test_loss + 1e-8)
            scheduler.step(epr_value)
        else:
            scheduler.step()

        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_accuracy,
                'config': model_config,
                'version': 'v1.0.0',
                'metadata': {
                    'dataset': 'CIFAR-10',
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'use_fdt': use_fdt
                }
            }
            torch.save(checkpoint, output_dir / "best_model.pt")
            logger.info(f"Saved best model (acc: {test_accuracy:.2f}%)")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_accuracy,
                'config': model_config,
                'history': train_history,
                'version': 'v1.0.0'
            }
            torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch}.pt")

    # ========================================================================
    # 5. Save Final Model
    # ========================================================================
    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info(f"Best test accuracy: {best_accuracy:.2f}%")
    logger.info(f"Checkpoints saved to: {output_dir}")
    logger.info("=" * 80)

    # Save final checkpoint
    final_checkpoint = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': test_accuracy,
        'config': model_config,
        'history': train_history,
        'version': 'v1.0.0',
        'metadata': {
            'dataset': 'CIFAR-10',
            'best_accuracy': best_accuracy,
            'final_accuracy': test_accuracy
        }
    }
    torch.save(final_checkpoint, output_dir / "final_model.pt")

    logger.info("\nNext steps:")
    logger.info("1. Export model for deployment:")
    logger.info(f"   python deploy/export_model.py --checkpoint {output_dir}/best_model.pt --output-dir deploy/models --compress")
    logger.info("2. Deploy model:")
    logger.info("   cd deploy && ./deploy.sh deploy-local")

    return model, train_history, best_accuracy


def main():
    parser = argparse.ArgumentParser(description="Train TFAN on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--no-fdt", action="store_true", help="Disable FDT scheduler")

    args = parser.parse_args()

    train_cifar10(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        output_dir=args.output_dir,
        use_fdt=not args.no_fdt
    )


if __name__ == "__main__":
    main()
