"""
MAML (Model-Agnostic Meta-Learning) implementation for TFAN.
Enables few-shot learning and rapid adaptation to new tasks.

Based on: Finn et al. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" (2017)
"""
import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
import json
import copy
import time

from .trainer import FDTScheduler
from .io import write_json


class MAMLTrainer:
    """
    MAML meta-learning trainer with FDT integration.

    Supports both first-order MAML (FOMAML) and full second-order MAML.
    Integrates with existing FDT scheduler for homeostatic meta-learning.
    """

    def __init__(self,
                 model: nn.Module,
                 meta_optimizer: Optimizer,
                 inner_lr: float = 0.01,
                 num_inner_steps: int = 5,
                 first_order: bool = False,
                 use_fdt_meta: bool = True,
                 target_meta_epr_cv: float = 0.15,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 log_dir: str = "logs/meta"):
        """
        Args:
            model: Base model to meta-learn
            meta_optimizer: Optimizer for meta-updates (outer loop)
            inner_lr: Learning rate for task adaptation (inner loop)
            num_inner_steps: Number of gradient steps for adaptation
            first_order: Use first-order approximation (FOMAML) for speed
            use_fdt_meta: Apply FDT to meta-learning process
            target_meta_epr_cv: Target EPR-CV for meta-learning
            device: Training device
            log_dir: Directory for meta-learning logs
        """
        self.model = model.to(device)
        self.meta_optimizer = meta_optimizer
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.first_order = first_order
        self.device = device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # FDT for meta-learning
        self.use_fdt_meta = use_fdt_meta
        if use_fdt_meta:
            self.meta_scheduler = FDTScheduler(
                optimizer=meta_optimizer,
                target_epr_cv=target_meta_epr_cv,
                adaptive_gains=True
            )
        else:
            self.meta_scheduler = None

        # Meta-learning statistics
        self.meta_step = 0
        self.meta_train_losses = []
        self.meta_val_losses = []
        self.adaptation_speeds = []

    def inner_loop(self,
                   task_model: nn.Module,
                   support_data: List[Tuple[torch.Tensor, torch.Tensor]],
                   criterion: nn.Module,
                   create_graph: bool = True) -> nn.Module:
        """
        Perform inner loop adaptation on support set.

        Args:
            task_model: Model copy for this task
            support_data: List of (input, target) tuples for support set
            criterion: Loss function
            create_graph: Whether to create computation graph (for 2nd order)

        Returns:
            Adapted model
        """
        # Create optimizer for inner loop (simple SGD)
        inner_optimizer = torch.optim.SGD(task_model.parameters(), lr=self.inner_lr)

        # Adaptation steps on support set
        for step in range(self.num_inner_steps):
            inner_optimizer.zero_grad()

            # Compute loss on support set
            total_loss = 0.0
            for inputs, targets in support_data:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = task_model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss

            # Average loss across support examples
            total_loss = total_loss / len(support_data)

            # Backward pass
            total_loss.backward(create_graph=create_graph and not self.first_order)

            # Update task-specific parameters
            inner_optimizer.step()

        return task_model

    def meta_train_step(self,
                        task_batch: List[Dict],
                        criterion: nn.Module) -> Dict[str, float]:
        """
        Single meta-training step on a batch of tasks.

        Args:
            task_batch: List of task dicts, each containing:
                - 'support': List of (input, target) tuples
                - 'query': List of (input, target) tuples
            criterion: Loss function

        Returns:
            Dict with meta-training statistics
        """
        self.meta_optimizer.zero_grad()

        meta_loss = 0.0
        task_losses = []

        for task in task_batch:
            # Create task-specific model copy
            task_model = copy.deepcopy(self.model)

            # Inner loop: adapt to support set
            adapted_model = self.inner_loop(
                task_model,
                task['support'],
                criterion,
                create_graph=not self.first_order
            )

            # Outer loop: evaluate on query set
            query_loss = 0.0
            for inputs, targets in task['query']:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = adapted_model(inputs)
                loss = criterion(outputs, targets)
                query_loss += loss

            # Average query loss
            query_loss = query_loss / len(task['query'])
            task_losses.append(query_loss.item())

            # Accumulate meta-loss
            meta_loss += query_loss

        # Average meta-loss across tasks
        meta_loss = meta_loss / len(task_batch)

        # Meta-backward pass
        meta_loss.backward()

        # Meta-update
        self.meta_optimizer.step()

        # Update FDT scheduler if enabled
        if self.meta_scheduler is not None:
            fdt_stats = self.meta_scheduler.step(meta_loss.item())
        else:
            fdt_stats = {}

        self.meta_step += 1
        self.meta_train_losses.append(meta_loss.item())

        stats = {
            "meta_loss": meta_loss.item(),
            "task_losses_mean": sum(task_losses) / len(task_losses),
            "task_losses_std": torch.std(torch.tensor(task_losses)).item(),
            "meta_step": self.meta_step,
        }
        stats.update(fdt_stats)

        return stats

    @torch.no_grad()
    def meta_validate(self,
                      val_tasks: List[Dict],
                      criterion: nn.Module) -> Dict[str, float]:
        """
        Validate on held-out tasks.

        Args:
            val_tasks: List of validation tasks
            criterion: Loss function

        Returns:
            Validation statistics
        """
        val_losses = []
        adaptation_times = []

        for task in val_tasks:
            # Create task-specific model
            task_model = copy.deepcopy(self.model)
            task_model.train()  # Need gradients for inner loop

            # Time adaptation
            t0 = time.time()

            # Adapt to support set
            with torch.enable_grad():
                adapted_model = self.inner_loop(
                    task_model,
                    task['support'],
                    criterion,
                    create_graph=False  # No gradients needed in validation
                )

            adaptation_time = time.time() - t0
            adaptation_times.append(adaptation_time)

            # Evaluate on query set
            adapted_model.eval()
            query_loss = 0.0
            for inputs, targets in task['query']:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = adapted_model(inputs)
                loss = criterion(outputs, targets)
                query_loss += loss.item()

            query_loss = query_loss / len(task['query'])
            val_losses.append(query_loss)

        self.meta_val_losses.append(sum(val_losses) / len(val_losses))

        return {
            "val_loss": sum(val_losses) / len(val_losses),
            "val_loss_std": torch.std(torch.tensor(val_losses)).item(),
            "avg_adaptation_time": sum(adaptation_times) / len(adaptation_times),
            "num_val_tasks": len(val_tasks)
        }

    def meta_train(self,
                   meta_train_loader,
                   meta_val_loader,
                   criterion: nn.Module,
                   num_meta_epochs: int,
                   tasks_per_batch: int = 4,
                   eval_every: int = 10) -> Dict:
        """
        Main meta-training loop.

        Args:
            meta_train_loader: DataLoader yielding task batches
            meta_val_loader: DataLoader for validation tasks
            criterion: Loss function
            num_meta_epochs: Number of meta-training epochs
            tasks_per_batch: Number of tasks per meta-batch
            eval_every: Evaluate every N meta-steps

        Returns:
            Meta-training history
        """
        print(f"=== Starting MAML Meta-Training ===")
        print(f"Meta-epochs: {num_meta_epochs}")
        print(f"Tasks per batch: {tasks_per_batch}")
        print(f"Inner LR: {self.inner_lr}")
        print(f"Inner steps: {self.num_inner_steps}")
        print(f"First-order: {self.first_order}")
        print(f"FDT meta: {self.use_fdt_meta}")
        print()

        for meta_epoch in range(num_meta_epochs):
            epoch_stats = []

            for task_batch_idx, task_batch in enumerate(meta_train_loader):
                # Ensure we have the right batch size
                if len(task_batch) != tasks_per_batch:
                    continue

                # Meta-training step
                stats = self.meta_train_step(task_batch, criterion)
                epoch_stats.append(stats)

                # Validation
                if self.meta_step % eval_every == 0:
                    val_stats = self.meta_validate(
                        list(meta_val_loader)[:20],  # Use first 20 val tasks
                        criterion
                    )

                    print(f"Meta-step {self.meta_step} | "
                          f"Train: {stats['meta_loss']:.4f} | "
                          f"Val: {val_stats['val_loss']:.4f} | "
                          f"Adapt time: {val_stats['avg_adaptation_time']*1000:.1f}ms")

                    if self.meta_scheduler is not None:
                        print(f"  EPR-CV: {stats.get('epr_cv', 0):.4f} | "
                              f"Meta-LR: {stats.get('lr', 0):.2e}")

            # Epoch summary
            avg_meta_loss = sum(s['meta_loss'] for s in epoch_stats) / len(epoch_stats)
            print(f"\nMeta-Epoch {meta_epoch+1}/{num_meta_epochs} | "
                  f"Avg Meta-Loss: {avg_meta_loss:.4f}\n")

        # Final evaluation
        print("=== Final Meta-Validation ===")
        final_val_stats = self.meta_validate(
            list(meta_val_loader),
            criterion
        )
        print(f"Final Val Loss: {final_val_stats['val_loss']:.4f}")
        print(f"Adaptation Time: {final_val_stats['avg_adaptation_time']*1000:.1f}ms")

        # Save history
        history = {
            "meta_train_losses": self.meta_train_losses,
            "meta_val_losses": self.meta_val_losses,
            "final_val_stats": final_val_stats,
            "config": {
                "inner_lr": self.inner_lr,
                "num_inner_steps": self.num_inner_steps,
                "first_order": self.first_order,
                "use_fdt_meta": self.use_fdt_meta
            }
        }

        if self.meta_scheduler is not None:
            history.update({
                "meta_epr_history": self.meta_scheduler.epr_history,
                "meta_lr_history": self.meta_scheduler.lr_history,
                "final_meta_epr_cv": self.meta_scheduler.epr_monitor.cv()
            })

        history_path = self.log_dir / f"meta_history_step{self.meta_step}.json"
        write_json(history_path, history)

        return history

    def fast_adapt(self,
                   support_data: List[Tuple[torch.Tensor, torch.Tensor]],
                   criterion: nn.Module,
                   num_steps: Optional[int] = None) -> nn.Module:
        """
        Quickly adapt to a new task using the meta-learned initialization.

        Args:
            support_data: Support set for new task
            criterion: Loss function
            num_steps: Number of adaptation steps (defaults to self.num_inner_steps)

        Returns:
            Adapted model ready for inference
        """
        num_steps = num_steps or self.num_inner_steps

        # Create fresh copy of meta-learned model
        adapted_model = copy.deepcopy(self.model)

        # Adapt to new task
        adapted_model = self.inner_loop(
            adapted_model,
            support_data,
            criterion,
            create_graph=False
        )

        adapted_model.eval()
        return adapted_model

    def save_meta_checkpoint(self, path: str):
        """Save meta-learned model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "meta_optimizer_state_dict": self.meta_optimizer.state_dict(),
            "meta_step": self.meta_step,
            "config": {
                "inner_lr": self.inner_lr,
                "num_inner_steps": self.num_inner_steps,
                "first_order": self.first_order
            }
        }

        if self.meta_scheduler is not None:
            checkpoint["meta_scheduler_state_dict"] = self.meta_scheduler.state_dict()

        torch.save(checkpoint, path)
        print(f"Meta-checkpoint saved to: {path}")

    def load_meta_checkpoint(self, path: str):
        """Load meta-learned model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.meta_optimizer.load_state_dict(checkpoint["meta_optimizer_state_dict"])
        self.meta_step = checkpoint["meta_step"]

        if self.meta_scheduler is not None and "meta_scheduler_state_dict" in checkpoint:
            self.meta_scheduler.load_state_dict(checkpoint["meta_scheduler_state_dict"])

        print(f"Meta-checkpoint loaded from: {path}")
        print(f"Meta-step: {self.meta_step}")


def compute_adaptation_metrics(model_before: nn.Module,
                               model_after: nn.Module,
                               test_data: List[Tuple[torch.Tensor, torch.Tensor]],
                               criterion: nn.Module,
                               device: str = "cpu") -> Dict[str, float]:
    """
    Compute metrics comparing pre and post adaptation.

    Args:
        model_before: Model before adaptation
        model_after: Model after adaptation
        test_data: Test data to evaluate on
        criterion: Loss function
        device: Device for computation

    Returns:
        Dict with before/after metrics and improvement
    """
    model_before.eval()
    model_after.eval()

    loss_before = 0.0
    loss_after = 0.0

    with torch.no_grad():
        for inputs, targets in test_data:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Before adaptation
            outputs_before = model_before(inputs)
            loss_before += criterion(outputs_before, targets).item()

            # After adaptation
            outputs_after = model_after(inputs)
            loss_after += criterion(outputs_after, targets).item()

    loss_before /= len(test_data)
    loss_after /= len(test_data)

    improvement = (loss_before - loss_after) / loss_before * 100 if loss_before > 0 else 0.0

    return {
        "loss_before_adaptation": loss_before,
        "loss_after_adaptation": loss_after,
        "improvement_percent": improvement
    }
