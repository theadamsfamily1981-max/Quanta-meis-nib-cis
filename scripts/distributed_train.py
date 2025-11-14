"""
Distributed training script for TFAN using PyTorch DDP.
Supports multi-GPU and multi-node training.

Usage:
    # Single node, multi-GPU
    torchrun --nproc_per_node=4 scripts/distributed_train.py

    # Multi-node
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=<addr> scripts/distributed_train.py
"""
import torch
import torch.nn as nn
import argparse
import os
from pathlib import Path

from tfan.distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    print_distributed_info,
    DistributedTFANTrainer,
    create_distributed_dataloaders,
    benchmark_communication
)
from tfan.datasets import create_dataloaders


class SimpleTFANModel(nn.Module):
    """Simple TFAN model for distributed training demo."""
    def __init__(self, input_dim: int = 512, hidden_dim: int = 1024, output_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def distributed_train(rank: int,
                     world_size: int,
                     num_epochs: int = 10,
                     batch_size: int = 32,
                     learning_rate: float = 1e-4):
    """
    Distributed training function.

    Args:
        rank: Process rank
        world_size: Total number of processes
        num_epochs: Number of training epochs
        batch_size: Batch size per GPU
        learning_rate: Learning rate
    """
    # Setup distributed
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    setup_distributed(rank, world_size, backend=backend)

    # Print distributed info
    if is_main_process():
        print_distributed_info()

    # Benchmark communication
    if torch.cuda.is_available():
        benchmark_communication()

    # Create model
    model = SimpleTFANModel(input_dim=512, hidden_dim=1024, output_dim=512)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Create distributed trainer
    trainer = DistributedTFANTrainer(
        model=model,
        optimizer=optimizer,
        rank=rank,
        world_size=world_size,
        use_fdt=True,
        target_epr_cv=0.15,
        sync_bn=True
    )

    # Create datasets
    train_dataset, val_dataset = create_dataloaders(
        dataset_type='long_sequence',
        seq_len=1024,
        batch_size=batch_size,
        num_train=1000,
        num_val=200
    )

    # Create distributed dataloaders
    train_loader, val_loader = create_distributed_dataloaders(
        dataset_train=train_dataset.dataset,
        dataset_val=val_dataset.dataset,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        num_workers=4
    )

    # Training loop
    criterion = nn.MSELoss()

    if is_main_process():
        print(f"\n=== Starting Distributed Training ===")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size per GPU: {batch_size}")
        print(f"Global batch size: {batch_size * world_size}")
        print(f"World size: {world_size}\n")

    for epoch in range(num_epochs):
        # Set epoch for distributed sampler
        train_loader.sampler.set_epoch(epoch)

        # Train
        train_loss = trainer.train_epoch(train_loader, criterion)

        # Validate
        val_loss = trainer.validate(val_loader, criterion)

        # Print stats (only on main process)
        if is_main_process():
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            if trainer.scheduler is not None:
                epr_cv = trainer.scheduler.epr_monitor.cv()
                lr = trainer.scheduler.current_lr
                print(f"  EPR-CV: {epr_cv:.4f} | LR: {lr:.2e}")

    # Save checkpoint (only on main process)
    if is_main_process():
        checkpoint_path = "checkpoints/distributed_final.pt"
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(checkpoint_path)

        print(f"\n=== Training Complete ===")
        print(f"Checkpoint saved to: {checkpoint_path}")

    # Cleanup
    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="Distributed TFAN training")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")

    args = parser.parse_args()

    # Get rank and world size from environment (set by torchrun)
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size == 1:
        print("Warning: Running in single-process mode. Use 'torchrun' for distributed training.")

    distributed_train(
        rank=rank,
        world_size=world_size,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )


if __name__ == "__main__":
    main()
