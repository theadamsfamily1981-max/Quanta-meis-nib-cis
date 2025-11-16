#!/usr/bin/env python
"""
Benchmark FSDP vs DDP for Multi-GPU Training

Compares Fully Sharded Data Parallel (FSDP) with Distributed Data Parallel (DDP)
for scaling 7B parameter models across multiple GPUs.

Hard gate:
- ≥1.6× speedup (FSDP vs DDP on 4 GPUs, 7B model, batch=2/GPU)

Usage:
    # Single GPU baseline
    python benchmarks/bench_fsdp_vs_ddp.py --gpus 1 --model-size 7b

    # DDP on 4 GPUs
    torchrun --nproc_per_node=4 benchmarks/bench_fsdp_vs_ddp.py \\
        --mode ddp --model-size 7b --batch-size 2

    # FSDP on 4 GPUs
    torchrun --nproc_per_node=4 benchmarks/bench_fsdp_vs_ddp.py \\
        --mode fsdp --model-size 7b --batch-size 2

    # Compare both (requires 4 GPUs)
    ./scripts/run_fsdp_benchmark.sh
"""

import argparse
import json
import time
import os
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Add tfan to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tfan.distributed.fsdp_orchestrator import FSDPOrchestrator, FSDPConfig


class MockTransformer(nn.Module):
    """
    Mock transformer model for benchmarking.

    Approximates structure of 7B/13B models:
    - 7B:  32 layers × 4096 dim × 32 heads (~6.7B params)
    - 13B: 40 layers × 5120 dim × 40 heads (~13B params)
    """

    def __init__(
        self,
        num_layers: int = 32,
        hidden_dim: int = 4096,
        num_heads: int = 32,
        vocab_size: int = 50257,
        max_seq_len: int = 2048
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Embedding
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])

        # Output head
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.embeddings(input_ids)  # [batch, seq_len, hidden_dim]

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits


class TransformerBlock(nn.Module):
    """Single transformer layer."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()

        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Attention
        x_norm = self.ln_1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.ln_2(x))

        return x


def get_model_config(model_size: str) -> Dict:
    """Get model configuration by size."""
    configs = {
        "1b": {"num_layers": 24, "hidden_dim": 2048, "num_heads": 16},
        "3b": {"num_layers": 28, "hidden_dim": 3072, "num_heads": 24},
        "7b": {"num_layers": 32, "hidden_dim": 4096, "num_heads": 32},
        "13b": {"num_layers": 40, "hidden_dim": 5120, "num_heads": 40},
    }

    return configs.get(model_size, configs["7b"])


def benchmark_training_step(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    num_iterations: int = 20,
    device: str = "cuda"
) -> Dict[str, float]:
    """Benchmark training throughput."""
    # Create dummy batch
    batch = {
        "input_ids": torch.randint(0, 50257, (batch_size, seq_len), device=device)
    }

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup
    for _ in range(5):
        logits = model(batch["input_ids"])
        loss = logits.mean()  # Dummy loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Synchronize
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    peak_mem = 0

    for _ in range(num_iterations):
        logits = model(batch["input_ids"])
        loss = logits.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if torch.cuda.is_available():
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated())

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.time() - start
    throughput = num_iterations * batch_size / elapsed  # samples/sec

    return {
        "time_per_step_ms": (elapsed / num_iterations) * 1000,
        "throughput_samples_per_sec": throughput,
        "peak_memory_gb": peak_mem / 1e9 if torch.cuda.is_available() else 0,
    }


def run_ddp_benchmark(
    model_size: str,
    batch_size: int,
    seq_len: int,
    num_iterations: int
) -> Dict:
    """Run DDP benchmark."""
    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    # Create model
    config = get_model_config(model_size)
    model = MockTransformer(**config).to(device)

    # Wrap with DDP
    model = DDP(model, device_ids=[rank])

    # Benchmark
    stats = benchmark_training_step(
        model, batch_size, seq_len, num_iterations, device
    )

    # Gather stats from all ranks
    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        stats["num_params"] = num_params
        stats["world_size"] = world_size
        stats["mode"] = "ddp"

    dist.barrier()
    dist.destroy_process_group()

    return stats


def run_fsdp_benchmark(
    model_size: str,
    batch_size: int,
    seq_len: int,
    num_iterations: int,
    cpu_offload: bool = False,
    activation_checkpointing: bool = False
) -> Dict:
    """Run FSDP benchmark."""
    # Create model
    config = get_model_config(model_size)
    model = MockTransformer(**config)

    # Wrap with FSDP
    fsdp_config = FSDPConfig(
        sharding_strategy="full",
        mixed_precision="bf16",
        cpu_offload=cpu_offload,
        activation_checkpointing=activation_checkpointing,
        transformer_layer_cls=TransformerBlock,
    )

    orchestrator = FSDPOrchestrator(model, config=fsdp_config)
    model = orchestrator.model

    # Benchmark
    def forward_fn(model, batch):
        return model(batch["input_ids"])

    def loss_fn(outputs, batch):
        return outputs.mean()

    # Warmup
    for _ in range(5):
        batch = {
            "input_ids": torch.randint(
                0, 50257, (batch_size, seq_len), device=orchestrator.device
            )
        }
        orchestrator.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        orchestrator.step(batch, forward_fn, loss_fn)

    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    peak_mem = 0

    for _ in range(num_iterations):
        batch = {
            "input_ids": torch.randint(
                0, 50257, (batch_size, seq_len), device=orchestrator.device
            )
        }
        orchestrator.step(batch, forward_fn, loss_fn)

        if torch.cuda.is_available():
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated())

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start
    throughput = num_iterations * batch_size * orchestrator.world_size / elapsed

    stats = {
        "time_per_step_ms": (elapsed / num_iterations) * 1000,
        "throughput_samples_per_sec": throughput,
        "peak_memory_gb": peak_mem / 1e9 if torch.cuda.is_available() else 0,
        "num_params": orchestrator.stats["num_params"],
        "world_size": orchestrator.world_size,
        "mode": "fsdp",
    }

    orchestrator.cleanup()

    return stats


def main():
    parser = argparse.ArgumentParser(description="Benchmark FSDP vs DDP")
    parser.add_argument("--mode", type=str, choices=["ddp", "fsdp"], required=True,
                        help="Training mode")
    parser.add_argument("--model-size", type=str, default="7b",
                        choices=["1b", "3b", "7b", "13b"], help="Model size")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--iterations", type=int, default=20, help="Number of iterations")
    parser.add_argument("--cpu-offload", action="store_true", help="Enable CPU offload (FSDP only)")
    parser.add_argument("--activation-checkpointing", action="store_true",
                        help="Enable activation checkpointing (FSDP only)")
    parser.add_argument("--output", type=str, help="Output JSON file")

    args = parser.parse_args()

    # Run benchmark
    if args.mode == "ddp":
        stats = run_ddp_benchmark(
            args.model_size, args.batch_size, args.seq_len, args.iterations
        )
    else:
        stats = run_fsdp_benchmark(
            args.model_size, args.batch_size, args.seq_len, args.iterations,
            args.cpu_offload, args.activation_checkpointing
        )

    # Print results
    rank = dist.get_rank() if dist.is_initialized() else 0

    if rank == 0:
        print("=" * 60)
        print(f"{args.mode.upper()} Benchmark Results")
        print("=" * 60)
        print(f"Model size:        {args.model_size} ({stats.get('num_params', 0) / 1e9:.2f}B params)")
        print(f"Batch size:        {args.batch_size}")
        print(f"Sequence length:   {args.seq_len}")
        print(f"World size:        {stats.get('world_size', 1)}")
        print(f"\nPerformance:")
        print(f"  Time/step:       {stats['time_per_step_ms']:.2f} ms")
        print(f"  Throughput:      {stats['throughput_samples_per_sec']:.2f} samples/sec")
        print(f"  Peak memory:     {stats['peak_memory_gb']:.2f} GB")

        # Save to file
        if args.output:
            with open(args.output, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"\n✓ Saved results to {args.output}")


if __name__ == "__main__":
    main()
