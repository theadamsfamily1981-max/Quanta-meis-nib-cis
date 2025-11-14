"""
Visualization tools for TFAN interpretability.
Provides visualizations for topology, attention patterns, and training dynamics.
"""
import torch
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import json

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from .topo import compute_persistence_diagram, PersistenceLandscape


class TopologyVisualizer:
    """Visualize topological features and persistence diagrams."""

    @staticmethod
    def plot_persistence_diagram(pd: np.ndarray, save_path: str,
                                 title: str = "Persistence Diagram",
                                 max_dim: int = 2):
        """
        Plot persistence diagram.

        Args:
            pd: Persistence diagram (Nx3 array: [dimension, birth, death])
            save_path: Path to save figure
            title: Plot title
            max_dim: Maximum dimension to plot
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for visualization")

        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot diagonal
        max_val = pd[:, 2].max() if len(pd) > 0 else 1.0
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Diagonal')

        # Plot points by dimension
        colors = ['red', 'blue', 'green', 'purple']
        dim_names = ['H0 (Connected Components)', 'H1 (Loops)', 'H2 (Voids)', 'H3+']

        for dim in range(max_dim + 1):
            dim_points = pd[pd[:, 0] == dim]
            if len(dim_points) > 0:
                births = dim_points[:, 1]
                deaths = dim_points[:, 2]

                ax.scatter(births, deaths,
                          c=colors[min(dim, len(colors)-1)],
                          label=dim_names[min(dim, len(dim_names)-1)],
                          alpha=0.6, s=50)

        ax.set_xlabel('Birth', fontsize=12)
        ax.set_ylabel('Death', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Persistence diagram saved to: {save_path}")

    @staticmethod
    def plot_persistence_landscape(pl_vectors: List[np.ndarray],
                                   save_path: str,
                                   labels: Optional[List[str]] = None):
        """
        Plot persistence landscapes.

        Args:
            pl_vectors: List of persistence landscape vectors
            save_path: Path to save figure
            labels: Labels for each landscape
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for visualization")

        fig, ax = plt.subplots(figsize=(12, 6))

        for idx, pl_vec in enumerate(pl_vectors):
            label = labels[idx] if labels and idx < len(labels) else f"Landscape {idx}"
            x = np.arange(len(pl_vec))
            ax.plot(x, pl_vec, label=label, alpha=0.7)

        ax.set_xlabel('Index', fontsize=12)
        ax.set_ylabel('Landscape Value', fontsize=12)
        ax.set_title('Persistence Landscapes', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Persistence landscapes saved to: {save_path}")

    @staticmethod
    def plot_topological_evolution(pd_sequence: List[np.ndarray],
                                   save_path: str,
                                   metric: str = "total_persistence"):
        """
        Plot evolution of topological features over time.

        Args:
            pd_sequence: Sequence of persistence diagrams
            save_path: Path to save figure
            metric: Metric to plot ('total_persistence', 'num_features', 'max_persistence')
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for visualization")

        values = []

        for pd in pd_sequence:
            if metric == "total_persistence":
                # Sum of persistence values
                persistence_vals = pd[:, 2] - pd[:, 1]
                val = persistence_vals.sum()
            elif metric == "num_features":
                # Number of features
                val = len(pd)
            elif metric == "max_persistence":
                # Maximum persistence
                persistence_vals = pd[:, 2] - pd[:, 1]
                val = persistence_vals.max() if len(persistence_vals) > 0 else 0.0
            else:
                raise ValueError(f"Unknown metric: {metric}")

            values.append(val)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(values, linewidth=2)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'Topological Evolution: {metric.replace("_", " ").title()}', fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Topological evolution saved to: {save_path}")


class AttentionVisualizer:
    """Visualize attention patterns and sparsity."""

    @staticmethod
    def plot_attention_heatmap(attn_weights: torch.Tensor,
                               save_path: str,
                               head_idx: int = 0,
                               max_tokens: int = 100):
        """
        Plot attention heatmap for a specific head.

        Args:
            attn_weights: Attention weights [B, H, T_q, T_k]
            save_path: Path to save figure
            head_idx: Head index to visualize
            max_tokens: Maximum number of tokens to show
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for visualization")

        # Extract head and convert to numpy
        if attn_weights.dim() == 4:
            attn_head = attn_weights[0, head_idx].detach().cpu().numpy()
        else:
            attn_head = attn_weights.detach().cpu().numpy()

        # Limit size for visualization
        T = min(attn_head.shape[0], max_tokens)
        attn_head = attn_head[:T, :T]

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(attn_head, cmap='viridis', aspect='auto')

        ax.set_xlabel('Key Position', fontsize=12)
        ax.set_ylabel('Query Position', fontsize=12)
        ax.set_title(f'Attention Heatmap (Head {head_idx})', fontsize=14)

        plt.colorbar(im, ax=ax, label='Attention Weight')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Attention heatmap saved to: {save_path}")

    @staticmethod
    def plot_sparsity_pattern(mask: torch.Tensor, save_path: str,
                             head_idx: int = 0, max_tokens: int = 200):
        """
        Plot sparse attention mask pattern.

        Args:
            mask: Boolean mask [B, H, T, T]
            save_path: Path to save figure
            head_idx: Head index to visualize
            max_tokens: Maximum tokens to show
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for visualization")

        # Extract head
        if mask.dim() == 4:
            mask_head = mask[0, head_idx].detach().cpu().numpy()
        else:
            mask_head = mask.detach().cpu().numpy()

        # Limit size
        T = min(mask_head.shape[0], max_tokens)
        mask_head = mask_head[:T, :T].astype(float)

        # Compute sparsity
        sparsity = 1.0 - (mask_head.sum() / (T * T))

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(mask_head, cmap='binary', aspect='auto')

        ax.set_xlabel('Key Position', fontsize=12)
        ax.set_ylabel('Query Position', fontsize=12)
        ax.set_title(f'Sparse Attention Pattern (Head {head_idx})\nSparsity: {sparsity*100:.1f}%', fontsize=14)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Sparsity pattern saved to: {save_path}")

    @staticmethod
    def plot_attention_distribution(attn_weights: torch.Tensor, save_path: str):
        """
        Plot distribution of attention weights across all heads.

        Args:
            attn_weights: Attention weights [B, H, T, T]
            save_path: Path to save figure
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for visualization")

        # Flatten all attention weights
        attn_flat = attn_weights.flatten().detach().cpu().numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax1.hist(attn_flat, bins=50, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Attention Weight', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Attention Weight Distribution', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Log scale
        ax2.hist(attn_flat[attn_flat > 1e-6], bins=50, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Attention Weight', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Attention Weight Distribution (Log Scale)', fontsize=14)
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Attention distribution saved to: {save_path}")


class TrainingVisualizer:
    """Visualize training dynamics and FDT behavior."""

    @staticmethod
    def plot_training_curves(history: Dict, save_path: str):
        """
        Plot training and validation loss curves.

        Args:
            history: Training history dict from TFANTrainer
            save_path: Path to save figure
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for visualization")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss curves
        if "train_losses" in history:
            axes[0, 0].plot(history["train_losses"], label='Train Loss', linewidth=2)
        if "val_losses" in history:
            axes[0, 0].plot(history["val_losses"], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training Curves', fontsize=14)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # EPR-CV evolution
        if "epr_cv_history" in history:
            axes[0, 1].plot(history["epr_cv_history"], linewidth=2, color='orange')
            if "target_epr_cv" in history:
                axes[0, 1].axhline(y=history["target_epr_cv"], color='red',
                                  linestyle='--', label=f'Target ({history["target_epr_cv"]:.2f})')
            else:
                axes[0, 1].axhline(y=0.15, color='red', linestyle='--', label='Target (0.15)')
            axes[0, 1].set_xlabel('Step', fontsize=12)
            axes[0, 1].set_ylabel('EPR-CV', fontsize=12)
            axes[0, 1].set_title('EPR-CV Evolution', fontsize=14)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Learning rate schedule
        if "lr_history" in history:
            axes[1, 0].plot(history["lr_history"], linewidth=2, color='green')
            axes[1, 0].set_xlabel('Step', fontsize=12)
            axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
            axes[1, 0].set_title('Learning Rate Schedule', fontsize=14)
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)

        # Temperature schedule
        if "temp_history" in history:
            axes[1, 1].plot(history["temp_history"], linewidth=2, color='purple')
            axes[1, 1].set_xlabel('Step', fontsize=12)
            axes[1, 1].set_ylabel('Temperature', fontsize=12)
            axes[1, 1].set_title('Temperature Schedule', fontsize=14)
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Training curves saved to: {save_path}")

    @staticmethod
    def plot_fdt_control_signals(history: Dict, save_path: str):
        """
        Plot FDT PID control signals over time.

        Args:
            history: Training history with EPR, LR, temp
            save_path: Path to save figure
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for visualization")

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # EPR with smoothing
        if "epr_history" in history:
            axes[0].plot(history["epr_history"], label='Raw EPR', alpha=0.5)
            if "smoothed_epr" in history:
                # Reconstruct smoothed EPR history if available
                axes[0].plot(range(len(history["epr_history"])),
                           [history["smoothed_epr"]] * len(history["epr_history"]),
                           label='Final Smoothed EPR', linestyle='--')
            axes[0].set_ylabel('EPR', fontsize=12)
            axes[0].set_title('Error Prediction Rate (EPR)', fontsize=14)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

        # Learning rate adaptation
        if "lr_history" in history:
            axes[1].plot(history["lr_history"], color='green', linewidth=2)
            axes[1].set_ylabel('Learning Rate', fontsize=12)
            axes[1].set_title('Adaptive Learning Rate (PID Control)', fontsize=14)
            axes[1].set_yscale('log')
            axes[1].grid(True, alpha=0.3)

        # Temperature adaptation
        if "temp_history" in history:
            axes[2].plot(history["temp_history"], color='purple', linewidth=2)
            axes[2].set_ylabel('Temperature', fontsize=12)
            axes[2].set_xlabel('Training Step', fontsize=12)
            axes[2].set_title('Adaptive Temperature (PID Control)', fontsize=14)
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"FDT control signals saved to: {save_path}")


def generate_all_visualizations(checkpoint_dir: str, output_dir: str):
    """
    Generate all standard visualizations from a checkpoint.

    Args:
        checkpoint_dir: Directory containing training checkpoints and history
        output_dir: Directory to save visualizations
    """
    checkpoint_path = Path(checkpoint_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating visualizations from {checkpoint_dir}...")

    # Load training history if available
    history_files = list(checkpoint_path.glob("training_history_*.json"))
    if history_files:
        latest_history = max(history_files, key=lambda p: p.stat().st_mtime)

        with open(latest_history) as f:
            history = json.load(f)

        # Training curves
        TrainingVisualizer.plot_training_curves(
            history,
            str(output_path / "training_curves.png")
        )

        # FDT control signals
        if "epr_history" in history:
            TrainingVisualizer.plot_fdt_control_signals(
                history,
                str(output_path / "fdt_control.png")
            )

    print("Visualization generation complete!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m tfan.viz <checkpoint_dir> <output_dir>")
        sys.exit(1)

    checkpoint_dir = sys.argv[1]
    output_dir = sys.argv[2]

    generate_all_visualizations(checkpoint_dir, output_dir)
