"""
Distributed training infrastructure for TFAN using PyTorch DDP.
Enables multi-GPU and multi-node training with gradient synchronization.
"""
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import Optimizer
from typing import Optional, Dict, Tuple
import os
import socket
from pathlib import Path
import time

from .trainer import FDTScheduler, TFANTrainer


def setup_distributed(rank: int, world_size: int,
                     backend: str = "nccl",
                     master_addr: str = "localhost",
                     master_port: str = "12355"):
    """
    Initialize distributed training process group.

    Args:
        rank: Rank of current process
        world_size: Total number of processes
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU)
        master_addr: Address of master node
        master_port: Port for communication
    """
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    # Initialize process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    # Set device for this process
    if backend == "nccl":
        torch.cuda.set_device(rank)

    print(f"[Rank {rank}] Initialized distributed training | "
          f"World size: {world_size} | Backend: {backend}")


def cleanup_distributed():
    """Clean up distributed process group."""
    dist.destroy_process_group()


def get_rank() -> int:
    """Get rank of current process."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    return get_rank() == 0


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def all_reduce(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """
    All-reduce operation across all processes.

    Args:
        tensor: Tensor to reduce
        op: Reduction operation

    Returns:
        Reduced tensor
    """
    if dist.is_initialized():
        dist.all_reduce(tensor, op=op)
    return tensor


def all_gather(tensor: torch.Tensor) -> list:
    """
    Gather tensors from all processes.

    Args:
        tensor: Tensor to gather

    Returns:
        List of tensors from all processes
    """
    if not dist.is_initialized():
        return [tensor]

    world_size = get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)

    return tensor_list


class DistributedTFANTrainer(TFANTrainer):
    """
    Distributed version of TFAN trainer using DDP.

    Extends TFANTrainer with multi-GPU/multi-node capabilities.
    """

    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 rank: int,
                 world_size: int,
                 device: Optional[str] = None,
                 use_fdt: bool = True,
                 target_epr_cv: float = 0.15,
                 log_dir: str = "logs",
                 sync_bn: bool = True,
                 gradient_as_bucket_view: bool = True):
        """
        Args:
            model: TFAN model
            optimizer: Optimizer
            rank: Process rank
            world_size: Total number of processes
            device: Device (auto-detected if None)
            use_fdt: Enable FDT scheduler
            target_epr_cv: Target EPR-CV
            log_dir: Log directory
            sync_bn: Synchronize batch normalization across GPUs
            gradient_as_bucket_view: Use bucket view for gradients (more efficient)
        """
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = f"cuda:{rank}"
            else:
                device = "cpu"

        self.rank = rank
        self.world_size = world_size

        # Convert BatchNorm to SyncBatchNorm if requested
        if sync_bn and torch.cuda.is_available():
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            print(f"[Rank {rank}] Converted to SyncBatchNorm")

        # Move model to device
        model = model.to(device)

        # Wrap with DDP
        self.ddp_model = DDP(
            model,
            device_ids=[rank] if torch.cuda.is_available() else None,
            output_device=rank if torch.cuda.is_available() else None,
            gradient_as_bucket_view=gradient_as_bucket_view,
            find_unused_parameters=False  # Set to True if needed
        )

        # Initialize parent trainer with wrapped model
        super().__init__(
            model=self.ddp_model.module,  # Access underlying model
            optimizer=optimizer,
            device=device,
            use_fdt=use_fdt,
            target_epr_cv=target_epr_cv,
            log_dir=log_dir
        )

        # Override model with DDP-wrapped version for training
        self._ddp_wrapped_model = self.ddp_model

    def train_epoch(self, train_loader: DataLoader, criterion: nn.Module) -> float:
        """Train for one epoch with distributed data parallel."""
        self._ddp_wrapped_model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass through DDP model
            outputs = self._ddp_wrapped_model(inputs)
            loss = criterion(outputs, targets)

            # Temperature scaling if FDT enabled
            if self.scheduler is not None:
                temp = self.scheduler.get_temperature()
                loss = loss / temp

            # Backward pass (gradients synchronized automatically by DDP)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1

        # Average loss across all processes
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

        # Synchronize loss across all ranks
        loss_tensor = torch.tensor(avg_loss, device=self.device)
        all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()

        self.train_losses.append(avg_loss)

        return avg_loss

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        """Validate on validation set (distributed)."""
        self._ddp_wrapped_model.eval()
        val_loss = 0.0
        num_batches = 0

        for inputs, targets in val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self._ddp_wrapped_model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            num_batches += 1

        avg_loss = val_loss / num_batches if num_batches > 0 else 0.0

        # Synchronize validation loss
        loss_tensor = torch.tensor(avg_loss, device=self.device)
        all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()

        self.val_losses.append(avg_loss)

        return avg_loss

    def save_checkpoint(self, path: str):
        """Save checkpoint (only on rank 0)."""
        if is_main_process():
            super().save_checkpoint(path)
            barrier()  # Wait for rank 0 to finish saving
        else:
            barrier()  # Wait for rank 0


def create_distributed_dataloaders(
        dataset_train,
        dataset_val,
        batch_size: int,
        world_size: int,
        rank: int,
        num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Create distributed data loaders with DistributedSampler.

    Args:
        dataset_train: Training dataset
        dataset_val: Validation dataset
        batch_size: Batch size per GPU
        world_size: Total number of processes
        rank: Current process rank
        num_workers: Number of data loading workers

    Returns:
        (train_loader, val_loader)
    """
    # Create distributed samplers
    train_sampler = DistributedSampler(
        dataset_train,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )

    val_sampler = DistributedSampler(
        dataset_val,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )

    # Create data loaders
    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def reduce_dict(input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Reduce dictionary of tensors across all processes.

    Args:
        input_dict: Dict of tensors to reduce

    Returns:
        Reduced dict
    """
    if not dist.is_initialized():
        return input_dict

    world_size = get_world_size()

    with torch.no_grad():
        keys = sorted(input_dict.keys())
        values = [input_dict[k] for k in keys]

        # Stack tensors
        values = torch.stack(values, dim=0)

        # Reduce
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values = values / world_size

        # Unpack
        reduced_dict = {k: v for k, v in zip(keys, values)}

    return reduced_dict


class DistributedMetrics:
    """
    Collect and aggregate metrics across distributed processes.
    """

    def __init__(self, device: str = "cuda"):
        """
        Args:
            device: Device for tensor operations
        """
        self.device = device
        self.metrics = {}

    def update(self, key: str, value: float, count: int = 1):
        """
        Update metric.

        Args:
            key: Metric name
            value: Metric value
            count: Number of samples
        """
        if key not in self.metrics:
            self.metrics[key] = {"sum": 0.0, "count": 0}

        self.metrics[key]["sum"] += value * count
        self.metrics[key]["count"] += count

    def compute(self) -> Dict[str, float]:
        """
        Compute and synchronize metrics across all processes.

        Returns:
            Dict of averaged metrics
        """
        if not self.metrics:
            return {}

        # Convert to tensors
        keys = sorted(self.metrics.keys())
        sums = torch.tensor([self.metrics[k]["sum"] for k in keys],
                           dtype=torch.float32, device=self.device)
        counts = torch.tensor([self.metrics[k]["count"] for k in keys],
                             dtype=torch.float32, device=self.device)

        # All-reduce
        if dist.is_initialized():
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)

        # Compute averages
        averages = (sums / counts).cpu().numpy()

        return {k: v for k, v in zip(keys, averages)}

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}


def get_hostname() -> str:
    """Get hostname of current machine."""
    return socket.gethostname()


def print_distributed_info():
    """Print distributed training information."""
    if not dist.is_initialized():
        print("Distributed training not initialized")
        return

    rank = get_rank()
    world_size = get_world_size()
    hostname = get_hostname()

    if rank == 0:
        print("=" * 60)
        print("Distributed Training Configuration")
        print("=" * 60)

    barrier()

    # Each process prints its info
    print(f"Rank {rank}/{world_size-1} | Host: {hostname} | "
          f"Device: {torch.cuda.get_device_name(rank) if torch.cuda.is_available() else 'CPU'}")

    barrier()

    if rank == 0:
        print("=" * 60)


def benchmark_communication(tensor_size: int = 1024 * 1024,
                           num_iters: int = 100) -> Dict[str, float]:
    """
    Benchmark distributed communication performance.

    Args:
        tensor_size: Size of tensor for communication
        num_iters: Number of iterations

    Returns:
        Dict with benchmark results
    """
    if not dist.is_initialized():
        return {"error": "Distributed not initialized"}

    rank = get_rank()
    world_size = get_world_size()

    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    tensor = torch.randn(tensor_size, device=device)

    # Warmup
    for _ in range(10):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Benchmark all_reduce
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    barrier()
    t0 = time.time()

    for _ in range(num_iters):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    barrier()
    elapsed = time.time() - t0

    # Compute bandwidth
    tensor_size_mb = tensor.numel() * tensor.element_size() / (1024 ** 2)
    # Each all_reduce transfers: (world_size - 1) / world_size * 2 * size
    # Approximate as 2 * size for simplicity
    total_data_mb = tensor_size_mb * 2 * num_iters
    bandwidth_mb_s = total_data_mb / elapsed

    results = {
        "num_iters": num_iters,
        "tensor_size_mb": tensor_size_mb,
        "elapsed_seconds": elapsed,
        "avg_latency_ms": (elapsed / num_iters) * 1000,
        "bandwidth_mb_s": bandwidth_mb_s,
        "world_size": world_size
    }

    if rank == 0:
        print(f"\n=== Communication Benchmark ===")
        print(f"Tensor size: {tensor_size_mb:.2f} MB")
        print(f"Iterations: {num_iters}")
        print(f"Avg latency: {results['avg_latency_ms']:.2f} ms")
        print(f"Bandwidth: {bandwidth_mb_s:.2f} MB/s")

    return results


if __name__ == "__main__":
    # Demo distributed utilities
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m tfan.distributed <rank> <world_size>")
        sys.exit(1)

    rank = int(sys.argv[1])
    world_size = int(sys.argv[2])

    # Setup distributed
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    setup_distributed(rank, world_size, backend=backend)

    # Print info
    print_distributed_info()

    # Benchmark communication
    if torch.cuda.is_available():
        benchmark_communication()

    # Cleanup
    cleanup_distributed()
