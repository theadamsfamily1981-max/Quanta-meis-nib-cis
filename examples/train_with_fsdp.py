#!/usr/bin/env python
"""
Example: Training with FSDP Orchestrator

Demonstrates how to use FSDPOrchestrator for multi-GPU training.

Usage:
    # Single GPU
    python examples/train_with_fsdp.py

    # Multi-GPU with FSDP
    torchrun --nproc_per_node=4 examples/train_with_fsdp.py \\
        --use-fsdp --epochs 10 --batch-size 2

    # With CPU offload and activation checkpointing
    torchrun --nproc_per_node=4 examples/train_with_fsdp.py \\
        --use-fsdp --cpu-offload --activation-checkpointing
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add tfan to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tfan.distributed import FSDPOrchestrator, FSDPConfig


class SimpleTransformer(nn.Module):
    """Simple transformer for demonstration."""

    def __init__(self, vocab_size: int = 10000, hidden_dim: int = 512, num_layers: int = 6):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=8, batch_first=True),
            num_layers=num_layers
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(input_ids)
        x = self.transformer(x)
        logits = self.lm_head(x)
        return logits


def create_dummy_dataset(num_samples: int = 1000, seq_len: int = 512, vocab_size: int = 10000):
    """Create dummy dataset for training."""
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    labels = torch.randint(0, vocab_size, (num_samples, seq_len))
    return TensorDataset(input_ids, labels)


def train_epoch(
    orchestrator: FSDPOrchestrator,
    dataloader: DataLoader,
    epoch: int
) -> dict:
    """Train for one epoch."""
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (input_ids, labels) in enumerate(dataloader):
        # Define forward and loss functions
        def forward_fn(model, batch):
            return model(batch["input_ids"])

        def loss_fn(outputs, batch):
            # Cross-entropy loss
            loss = nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                batch["labels"].view(-1)
            )
            return loss

        # Prepare batch
        batch = {
            "input_ids": input_ids,
            "labels": labels
        }

        # Training step
        stats = orchestrator.step(batch, forward_fn, loss_fn)

        total_loss += stats["loss"]
        num_batches += 1

        if batch_idx % 10 == 0 and orchestrator.rank == 0:
            print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {stats['loss']:.4f}")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    return {"avg_loss": avg_loss}


def main():
    parser = argparse.ArgumentParser(description="Train with FSDP")
    parser.add_argument("--use-fsdp", action="store_true", help="Use FSDP")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--cpu-offload", action="store_true", help="Enable CPU offload")
    parser.add_argument("--activation-checkpointing", action="store_true",
                        help="Enable activation checkpointing")
    parser.add_argument("--checkpoint-dir", type=str, default="artifacts/checkpoints",
                        help="Checkpoint directory")

    args = parser.parse_args()

    # Create model
    model = SimpleTransformer(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )

    # Print model stats
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params / 1e6:.2f}M parameters")

    # Wrap with FSDP if requested
    if args.use_fsdp:
        config = FSDPConfig(
            sharding_strategy="full",
            mixed_precision="bf16",
            cpu_offload=args.cpu_offload,
            activation_checkpointing=args.activation_checkpointing,
            transformer_layer_cls=nn.TransformerEncoderLayer,
        )

        orchestrator = FSDPOrchestrator(model, config=config)
        model = orchestrator.model

        # Create optimizer
        orchestrator.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        print(f"FSDP enabled:")
        print(f"  World size: {orchestrator.world_size}")
        print(f"  Rank: {orchestrator.rank}")
        print(f"  CPU offload: {args.cpu_offload}")
        print(f"  Activation checkpointing: {args.activation_checkpointing}")

    else:
        # Single GPU training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Create dummy orchestrator for uniform interface
        class DummyOrchestrator:
            def __init__(self, model, optimizer):
                self.model = model
                self.optimizer = optimizer
                self.rank = 0
                self.world_size = 1
                self.device = device

            def step(self, batch, forward_fn, loss_fn):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = forward_fn(self.model, batch)
                loss = loss_fn(outputs, batch)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                return {"loss": loss.item(), "grad_norm": 0.0}

        orchestrator = DummyOrchestrator(model, optimizer)

    # Create dataset and dataloader
    dataset = create_dummy_dataset(num_samples=1000, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        stats = train_epoch(orchestrator, dataloader, epoch)

        if orchestrator.rank == 0:
            print(f"Epoch {epoch} completed | Avg loss: {stats['avg_loss']:.4f}")

            # Save checkpoint
            if hasattr(orchestrator, 'save_checkpoint'):
                checkpoint_path = f"{args.checkpoint_dir}/checkpoint_epoch_{epoch}.pt"
                Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
                orchestrator.save_checkpoint(checkpoint_path, epoch=epoch)

    print("\nTraining complete!")

    # Cleanup
    if hasattr(orchestrator, 'cleanup'):
        orchestrator.cleanup()


if __name__ == "__main__":
    main()
