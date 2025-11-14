"""
TFAN Trainer with FDT homeostatic loop integration.
PID-controlled learning rate and temperature scheduling with EPR-CV monitoring.
"""
import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Optional, Dict, List
import numpy as np
from pathlib import Path
import json

from .fdt import PID, EPRCV
from .io import new_artifact, write_json


class FDTScheduler:
    """
    FDT (Feedback Dynamic Tuning) scheduler for learning rate and temperature.
    Uses PID controller to maintain EPR-CV within target range.
    """

    def __init__(self, optimizer: Optimizer,
                 initial_lr: float = 1e-4,
                 initial_temp: float = 1.0,
                 target_epr_cv: float = 0.15,
                 pid_kp: float = 0.25,
                 pid_ki: float = 0.03,
                 pid_kd: float = 0.08,
                 epr_window: int = 120):
        """
        Args:
            optimizer: PyTorch optimizer
            initial_lr: Initial learning rate
            initial_temp: Initial temperature
            target_epr_cv: Target EPR coefficient of variation
            pid_kp, pid_ki, pid_kd: PID controller gains
            epr_window: EPR-CV window size
        """
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.initial_temp = initial_temp
        self.target_epr_cv = target_epr_cv

        # PID controllers for LR and temperature
        self.lr_pid = PID(kp=pid_kp, ki=pid_ki, kd=pid_kd, clamp=(0.1, 2.0))
        self.temp_pid = PID(kp=pid_kp, ki=pid_ki, kd=pid_kd, clamp=(0.5, 2.0))

        # EPR-CV monitor
        self.epr_monitor = EPRCV(window=epr_window)

        # Current values
        self.current_lr = initial_lr
        self.current_temp = initial_temp

        # Logging
        self.epr_history = []
        self.lr_history = []
        self.temp_history = []

    def step(self, epr_value: float) -> Dict[str, float]:
        """
        Update learning rate and temperature based on EPR.

        Args:
            epr_value: Current EPR (e.g., validation loss or error rate)

        Returns:
            dict with updated values
        """
        # Update EPR monitor
        self.epr_monitor.push(epr_value)
        current_epr_cv = self.epr_monitor.cv()

        # Compute error from target
        error = current_epr_cv - self.target_epr_cv

        # Update LR via PID (reduce LR if EPR-CV too high)
        lr_scale = self.lr_pid.step(-error)
        self.current_lr = self.initial_lr * lr_scale

        # Update temperature via PID
        temp_scale = self.temp_pid.step(-error)
        self.current_temp = self.initial_temp * temp_scale

        # Apply to optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr

        # Log
        self.epr_history.append(epr_value)
        self.lr_history.append(self.current_lr)
        self.temp_history.append(self.current_temp)

        return {
            "epr": epr_value,
            "epr_cv": current_epr_cv,
            "lr": self.current_lr,
            "temperature": self.current_temp,
            "lr_scale": lr_scale,
            "temp_scale": temp_scale
        }

    def get_temperature(self) -> float:
        """Get current temperature for loss scaling."""
        return self.current_temp

    def state_dict(self) -> dict:
        """Get scheduler state for checkpointing."""
        return {
            "current_lr": self.current_lr,
            "current_temp": self.current_temp,
            "epr_history": self.epr_history,
            "lr_history": self.lr_history,
            "temp_history": self.temp_history,
            "epr_monitor_buf": list(self.epr_monitor.buf),
            "lr_pid_ei": self.lr_pid.ei,
            "lr_pid_prev_e": self.lr_pid.prev_e,
            "temp_pid_ei": self.temp_pid.ei,
            "temp_pid_prev_e": self.temp_pid.prev_e,
        }

    def load_state_dict(self, state_dict: dict):
        """Load scheduler state from checkpoint."""
        self.current_lr = state_dict["current_lr"]
        self.current_temp = state_dict["current_temp"]
        self.epr_history = state_dict["epr_history"]
        self.lr_history = state_dict["lr_history"]
        self.temp_history = state_dict["temp_history"]
        self.epr_monitor.buf = state_dict["epr_monitor_buf"]
        self.lr_pid.ei = state_dict["lr_pid_ei"]
        self.lr_pid.prev_e = state_dict["lr_pid_prev_e"]
        self.temp_pid.ei = state_dict["temp_pid_ei"]
        self.temp_pid.prev_e = state_dict["temp_pid_prev_e"]


class TFANTrainer:
    """
    Main training loop for TFAN with FDT homeostatic control.
    """

    def __init__(self, model: nn.Module, optimizer: Optimizer,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 use_fdt: bool = True,
                 target_epr_cv: float = 0.15,
                 log_dir: str = "logs"):
        """
        Args:
            model: TFAN model
            optimizer: PyTorch optimizer
            device: Training device
            use_fdt: Enable FDT scheduler
            target_epr_cv: Target EPR-CV for homeostasis
            log_dir: Directory for logs
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.use_fdt = use_fdt
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # FDT scheduler
        if use_fdt:
            self.scheduler = FDTScheduler(
                optimizer=optimizer,
                target_epr_cv=target_epr_cv
            )
        else:
            self.scheduler = None

        # Training stats
        self.epoch = 0
        self.global_step = 0
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader, criterion: nn.Module) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)

            # Temperature scaling if FDT enabled
            if self.scheduler is not None:
                temp = self.scheduler.get_temperature()
                loss = loss / temp

            # Backward pass
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        self.train_losses.append(avg_loss)

        return avg_loss

    @torch.no_grad()
    def validate(self, val_loader, criterion: nn.Module) -> float:
        """Validate on validation set."""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        for inputs, targets in val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            num_batches += 1

        avg_loss = val_loss / num_batches if num_batches > 0 else 0.0
        self.val_losses.append(avg_loss)

        return avg_loss

    def train(self, train_loader, val_loader, criterion: nn.Module,
              num_epochs: int, eval_every: int = 1) -> dict:
        """
        Main training loop.

        Returns:
            Training history dict
        """
        print(f"Training for {num_epochs} epochs...")
        print(f"FDT scheduler: {'enabled' if self.use_fdt else 'disabled'}")

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train
            train_loss = self.train_epoch(train_loader, criterion)

            # Validate
            if epoch % eval_every == 0:
                val_loss = self.validate(val_loader, criterion)

                # Update FDT scheduler with validation loss
                if self.scheduler is not None:
                    fdt_stats = self.scheduler.step(val_loss)

                    print(f"Epoch {epoch}/{num_epochs} | "
                          f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                          f"EPR-CV: {fdt_stats['epr_cv']:.4f} | "
                          f"LR: {fdt_stats['lr']:.2e} | "
                          f"Temp: {fdt_stats['temperature']:.3f}")
                else:
                    print(f"Epoch {epoch}/{num_epochs} | "
                          f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        # Save final stats
        history = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

        if self.scheduler is not None:
            history.update({
                "epr_history": self.scheduler.epr_history,
                "lr_history": self.scheduler.lr_history,
                "temp_history": self.scheduler.temp_history,
                "final_epr_cv": self.scheduler.epr_monitor.cv()
            })

        # Write to file
        history_path = self.log_dir / f"training_history_epoch{self.epoch}.json"
        write_json(history_path, history)

        return history

    def check_epr_cv_gate(self, num_windows: int = 3, threshold: float = 0.15) -> bool:
        """
        Check if EPR-CV gate is satisfied (≤ threshold sustained over windows).

        Args:
            num_windows: Number of eval windows to check
            threshold: EPR-CV threshold

        Returns:
            True if gate passed
        """
        if self.scheduler is None:
            return True  # FDT not enabled

        # Check last N validation losses
        if len(self.val_losses) < num_windows:
            return False

        recent_vals = self.val_losses[-num_windows:]

        # Compute CV for recent window
        if len(recent_vals) < 2:
            return False

        mean_val = np.mean(recent_vals)
        std_val = np.std(recent_vals)

        if mean_val < 1e-10:
            return True

        cv = std_val / mean_val

        return cv <= threshold
