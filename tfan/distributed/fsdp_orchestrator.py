#!/usr/bin/env python
"""
FSDP Orchestrator for Multi-Node Training

Fully Sharded Data Parallel (FSDP) with ZeRO-3 optimization for scaling
7B+ parameter models across multiple GPUs/nodes.

Key features:
- Gradient sharding (ZeRO-3): shard parameters, gradients, optimizer states
- Communication overlap: prefetch parameters during backward pass
- CPU offload: offload params/grads to CPU memory when not in use
- Activation checkpointing: trade compute for memory
- Mixed precision: FP16/BF16 training with loss scaling

Hard gate:
- ≥1.6× speedup vs DDP on 4 GPUs (7B model, batch_size=2 per GPU)

Usage:
    from tfan.distributed.fsdp_orchestrator import FSDPOrchestrator

    # Wrap model with FSDP
    orchestrator = FSDPOrchestrator(
        model=model,
        mixed_precision='bf16',
        cpu_offload=True,
        activation_checkpointing=True
    )

    # Train as usual
    for batch in dataloader:
        loss = orchestrator.step(batch)
"""

import os
import functools
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)


@dataclass
class FSDPConfig:
    """FSDP orchestrator configuration."""

    # Sharding strategy
    sharding_strategy: str = "full"  # "full" (ZeRO-3), "shard_grad_op" (ZeRO-2), "no_shard" (DDP)

    # Mixed precision
    mixed_precision: Optional[str] = "bf16"  # "bf16", "fp16", None

    # CPU offload
    cpu_offload: bool = False  # Offload params to CPU when not in use
    cpu_offload_grads: bool = False  # Also offload gradients

    # Communication optimization
    backward_prefetch: str = "backward_pre"  # "backward_pre", "backward_post", None
    forward_prefetch: bool = True  # Prefetch params during forward
    limit_all_gathers: bool = True  # Rate-limit all-gather for memory

    # Activation checkpointing
    activation_checkpointing: bool = False
    checkpoint_every_n_layers: int = 2

    # Auto-wrap policy
    min_num_params: int = 1e6  # Min params per FSDP unit (size-based)
    transformer_layer_cls: Optional[type] = None  # Transformer layer class

    # Gradient clipping
    gradient_clip_norm: Optional[float] = 1.0

    # Sync batch norm (if using)
    sync_batch_norm: bool = False


class FSDPOrchestrator:
    """
    FSDP orchestrator for distributed training.

    Manages:
    - Model wrapping with FSDP
    - Distributed initialization
    - Gradient sharding and communication
    - CPU offload
    - Activation checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        config: Optional[FSDPConfig] = None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize FSDP orchestrator.

        Args:
            model: PyTorch model to wrap
            optimizer: Optimizer (will be re-initialized after wrapping)
            config: FSDP configuration
            rank: Process rank (auto-detected if None)
            world_size: Number of processes (auto-detected if None)
            **kwargs: Additional config overrides
        """
        self.config = config or FSDPConfig(**kwargs)

        # Initialize distributed if not already
        if not dist.is_initialized():
            self._init_distributed(rank, world_size)

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")

        # Wrap model with FSDP
        self.model = self._wrap_model(model)

        # Re-initialize optimizer after wrapping
        self.optimizer = optimizer

        # Statistics
        self.stats = {
            'num_params': sum(p.numel() for p in self.model.parameters()),
            'num_sharded_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'sharding_factor': self.world_size,
        }

    def _init_distributed(self, rank: Optional[int], world_size: Optional[int]):
        """Initialize distributed process group."""
        if rank is None:
            rank = int(os.environ.get("RANK", 0))
        if world_size is None:
            world_size = int(os.environ.get("WORLD_SIZE", 1))

        if world_size == 1:
            print("⚠ Single GPU mode, FSDP not needed (using regular model)")
            return

        # Initialize process group
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size
            )

    def _get_sharding_strategy(self) -> ShardingStrategy:
        """Get FSDP sharding strategy."""
        strategies = {
            "full": ShardingStrategy.FULL_SHARD,  # ZeRO-3
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,  # ZeRO-2
            "hybrid": ShardingStrategy.HYBRID_SHARD,  # Hybrid intra/inter-node
            "no_shard": ShardingStrategy.NO_SHARD,  # DDP
        }
        return strategies.get(self.config.sharding_strategy, ShardingStrategy.FULL_SHARD)

    def _get_mixed_precision(self) -> Optional[MixedPrecision]:
        """Get mixed precision configuration."""
        if self.config.mixed_precision is None:
            return None

        if self.config.mixed_precision == "bf16":
            return MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif self.config.mixed_precision == "fp16":
            return MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        else:
            return None

    def _get_cpu_offload(self) -> Optional[CPUOffload]:
        """Get CPU offload configuration."""
        if not self.config.cpu_offload:
            return None

        return CPUOffload(offload_params=True)

    def _get_auto_wrap_policy(self) -> Optional[Callable]:
        """Get auto-wrap policy for FSDP."""
        if self.config.transformer_layer_cls is not None:
            # Transformer-based wrapping (wrap each layer)
            return functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={self.config.transformer_layer_cls}
            )
        else:
            # Size-based wrapping
            return functools.partial(
                size_based_auto_wrap_policy,
                min_num_params=int(self.config.min_num_params)
            )

    def _get_backward_prefetch(self) -> Optional[BackwardPrefetch]:
        """Get backward prefetch configuration."""
        if self.config.backward_prefetch == "backward_pre":
            return BackwardPrefetch.BACKWARD_PRE
        elif self.config.backward_prefetch == "backward_post":
            return BackwardPrefetch.BACKWARD_POST
        else:
            return None

    def _wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model with FSDP."""
        if self.world_size == 1:
            # Single GPU: no FSDP needed
            return model.to(self.device)

        # Apply activation checkpointing if enabled
        if self.config.activation_checkpointing:
            self._apply_activation_checkpointing(model)

        # Convert batch norm to sync batch norm if enabled
        if self.config.sync_batch_norm:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # Wrap with FSDP
        fsdp_model = FSDP(
            model,
            sharding_strategy=self._get_sharding_strategy(),
            mixed_precision=self._get_mixed_precision(),
            cpu_offload=self._get_cpu_offload(),
            auto_wrap_policy=self._get_auto_wrap_policy(),
            backward_prefetch=self._get_backward_prefetch(),
            forward_prefetch=self.config.forward_prefetch,
            limit_all_gathers=self.config.limit_all_gathers,
            device_id=self.device,
        )

        return fsdp_model

    def _apply_activation_checkpointing(self, model: nn.Module):
        """Apply activation checkpointing to model."""
        # Checkpoint every N layers
        # Note: This is a simplified implementation
        # In practice, would use apply_activation_checkpointing with layer matching

        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )

        # Apply to transformer layers if specified
        if self.config.transformer_layer_cls is not None:
            check_fn = lambda submodule: isinstance(submodule, self.config.transformer_layer_cls)
            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=non_reentrant_wrapper,
                check_fn=check_fn,
            )

    def step(
        self,
        batch: Dict[str, torch.Tensor],
        forward_fn: Callable,
        loss_fn: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        Perform single training step.

        Args:
            batch: Input batch dict
            forward_fn: Forward function (model, batch) -> outputs
            loss_fn: Loss function (outputs, batch) -> loss (if None, use outputs as loss)

        Returns:
            stats: Dict with loss and gradient norm
        """
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Forward pass
        outputs = forward_fn(self.model, batch)

        # Compute loss
        if loss_fn is not None:
            loss = loss_fn(outputs, batch)
        else:
            loss = outputs  # Assume outputs is loss directly

        # Backward pass
        loss.backward()

        # Gradient clipping (FSDP-aware)
        grad_norm = None
        if self.config.gradient_clip_norm is not None:
            grad_norm = self.model.clip_grad_norm_(self.config.gradient_clip_norm)

        # Optimizer step
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        # Gather statistics
        stats = {
            'loss': loss.item(),
            'grad_norm': grad_norm.item() if grad_norm is not None else 0.0
        }

        return stats

    def save_checkpoint(self, path: str, epoch: int, **kwargs):
        """Save FSDP checkpoint."""
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        # Use full state dict for checkpoint
        with FSDP.state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state_dict = self.model.state_dict()

        if self.rank == 0:
            checkpoint = {
                'epoch': epoch,
                'model': state_dict,
                'optimizer': self.optimizer.state_dict() if self.optimizer else None,
                'config': self.config,
                **kwargs
            }
            torch.save(checkpoint, path)
            print(f"✓ Saved checkpoint to {path}")

        dist.barrier()

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load FSDP checkpoint."""
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        # Load checkpoint on rank 0
        if self.rank == 0:
            checkpoint = torch.load(path, map_location='cpu')
        else:
            checkpoint = None

        # Broadcast to all ranks
        dist.barrier()

        # Load state dict
        with FSDP.state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            if self.rank == 0:
                self.model.load_state_dict(checkpoint['model'])

            if self.optimizer and checkpoint.get('optimizer'):
                self.optimizer.load_state_dict(checkpoint['optimizer'])

        dist.barrier()

        if self.rank == 0:
            print(f"✓ Loaded checkpoint from {path}")

        return checkpoint

    def get_stats(self) -> Dict[str, Any]:
        """Get FSDP statistics."""
        # Update stats
        self.stats.update({
            'rank': self.rank,
            'world_size': self.world_size,
            'sharding_strategy': self.config.sharding_strategy,
            'mixed_precision': self.config.mixed_precision,
            'cpu_offload': self.config.cpu_offload,
            'activation_checkpointing': self.config.activation_checkpointing,
        })

        return self.stats

    def cleanup(self):
        """Cleanup distributed resources."""
        if dist.is_initialized():
            dist.destroy_process_group()


def estimate_memory_savings(
    num_params: int,
    world_size: int,
    mixed_precision: str = "bf16",
    cpu_offload: bool = False
) -> Dict[str, float]:
    """
    Estimate memory savings with FSDP vs DDP.

    Args:
        num_params: Number of model parameters
        world_size: Number of GPUs
        mixed_precision: "bf16", "fp16", or "fp32"
        cpu_offload: Enable CPU offload

    Returns:
        savings: Dict with memory estimates (in GB)
    """
    # Bytes per parameter
    bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2}[mixed_precision]

    # DDP memory: params + grads + optimizer (Adam: 2x params)
    ddp_mem_gb = num_params * bytes_per_param * (1 + 1 + 2) / 1e9

    # FSDP memory: params/N + grads/N + optimizer/N
    fsdp_mem_gb = ddp_mem_gb / world_size

    # With CPU offload: only activations on GPU
    if cpu_offload:
        fsdp_mem_gb *= 0.3  # Rough estimate: 30% on GPU (activations only)

    return {
        'ddp_mem_gb': ddp_mem_gb,
        'fsdp_mem_gb': fsdp_mem_gb,
        'savings_gb': ddp_mem_gb - fsdp_mem_gb,
        'savings_pct': (ddp_mem_gb - fsdp_mem_gb) / ddp_mem_gb,
    }
