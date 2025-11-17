#!/usr/bin/env python
"""
Pareto Front Visualization Dashboard

Interactive dashboard for visualizing and analyzing Pareto fronts.

Features:
- 2D/3D scatter plots of Pareto fronts
- Parallel coordinates plot for all objectives
- Configuration comparison table
- Metrics summary

Usage:
    python dashboards/pareto_app.py --results artifacts/pareto/pareto_front.json
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plots will be skipped.")

try:
    from matplotlib.patches import Rectangle
    import matplotlib.gridspec as gridspec
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


class ParetoVisualizer:
    """Visualizer for Pareto fronts."""

    def __init__(self, results_path: str):
        """
        Initialize visualizer.

        Args:
            results_path: Path to pareto_front.json
        """
        self.results_path = Path(results_path)

        if not self.results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")

        # Load results
        with open(self.results_path, "r") as f:
            self.results = json.load(f)

        self.configs = self.results["configurations"]
        self.n_pareto_points = self.results["n_pareto_points"]
        self.hypervolume = self.results["hypervolume"]

        # Extract objectives
        self.objectives = np.array([c["objectives"] for c in self.configs])
        self.objective_names = ["Accuracy", "Latency (ms)", "EPR-CV", "Topo Gap", "Energy (W)"]

        # Negate accuracy for display
        self.objectives_display = self.objectives.copy()
        self.objectives_display[:, 0] = -self.objectives_display[:, 0]  # Convert back to positive

    def plot_2d_front(self, obj_x: int = 0, obj_y: int = 1, save_path: Optional[str] = None):
        """
        Plot 2D Pareto front.

        Args:
            obj_x: Index of x-axis objective
            obj_y: Index of y-axis objective
            save_path: Path to save plot
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available. Skipping 2D plot.")
            return

        plt.figure(figsize=(10, 6))

        plt.scatter(
            self.objectives_display[:, obj_x],
            self.objectives_display[:, obj_y],
            c=range(len(self.objectives)),
            cmap="viridis",
            s=100,
            alpha=0.7,
            edgecolors="black",
        )

        plt.xlabel(self.objective_names[obj_x], fontsize=12)
        plt.ylabel(self.objective_names[obj_y], fontsize=12)
        plt.title(f"Pareto Front: {self.objective_names[obj_x]} vs {self.objective_names[obj_y]}", fontsize=14)

        plt.colorbar(label="Configuration Index")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved 2D plot to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_parallel_coordinates(self, save_path: Optional[str] = None):
        """
        Plot parallel coordinates for all objectives.

        Args:
            save_path: Path to save plot
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available. Skipping parallel coordinates.")
            return

        fig, ax = plt.subplots(figsize=(14, 6))

        # Normalize objectives to [0, 1]
        objectives_norm = (self.objectives_display - self.objectives_display.min(axis=0)) / \
                         (self.objectives_display.max(axis=0) - self.objectives_display.min(axis=0) + 1e-8)

        n_objectives = objectives_norm.shape[1]
        x = np.arange(n_objectives)

        # Plot each configuration as a line
        for i, obj_values in enumerate(objectives_norm):
            ax.plot(x, obj_values, marker="o", alpha=0.5, label=f"Config {i}")

        ax.set_xticks(x)
        ax.set_xticklabels(self.objective_names, rotation=45, ha="right")
        ax.set_ylabel("Normalized Value", fontsize=12)
        ax.set_title("Parallel Coordinates Plot of Pareto Front", fontsize=14)
        ax.grid(True, alpha=0.3, axis="y")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved parallel coordinates to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_all_pairs(self, save_path: Optional[str] = None):
        """
        Plot pairwise objective scatter plots.

        Args:
            save_path: Path to save plot
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available. Skipping pairwise plots.")
            return

        n_obj = len(self.objective_names)
        fig, axes = plt.subplots(n_obj, n_obj, figsize=(16, 16))

        for i in range(n_obj):
            for j in range(n_obj):
                ax = axes[i, j]

                if i == j:
                    # Diagonal: histogram
                    ax.hist(self.objectives_display[:, i], bins=10, alpha=0.7, color="skyblue", edgecolor="black")
                    ax.set_ylabel("Count")
                else:
                    # Off-diagonal: scatter
                    ax.scatter(
                        self.objectives_display[:, j],
                        self.objectives_display[:, i],
                        alpha=0.6,
                        s=50,
                    )

                if i == n_obj - 1:
                    ax.set_xlabel(self.objective_names[j], fontsize=10)
                else:
                    ax.set_xticklabels([])

                if j == 0:
                    ax.set_ylabel(self.objective_names[i], fontsize=10)
                else:
                    ax.set_yticklabels([])

                ax.grid(True, alpha=0.2)

        plt.suptitle("Pairwise Objective Scatter Plots", fontsize=16, y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved pairwise plots to: {save_path}")
        else:
            plt.show()

        plt.close()

    def print_summary(self):
        """Print summary of Pareto front."""
        print("=" * 70)
        print("PARETO FRONT SUMMARY")
        print("=" * 70)
        print(f"Number of Pareto-optimal configurations: {self.n_pareto_points}")
        print(f"Hypervolume: {self.hypervolume:.2f}")
        print()

        print("Objective Statistics:")
        print("-" * 70)
        for i, name in enumerate(self.objective_names):
            values = self.objectives_display[:, i]
            print(f"{name:20s}: min={values.min():.4f}, max={values.max():.4f}, mean={values.mean():.4f}")
        print()

    def print_configurations(self, n_show: int = 5):
        """
        Print top configurations.

        Args:
            n_show: Number of configurations to show
        """
        print("Top Configurations:")
        print("-" * 70)

        for i, config in enumerate(self.configs[:n_show]):
            print(f"\nConfiguration {i + 1}:")
            print(f"  n_heads: {config['n_heads']}")
            print(f"  d_model: {config['d_model']}")
            print(f"  keep_ratio: {config['keep_ratio']:.3f}")
            print(f"  alpha: {config['alpha']:.3f}")
            print(f"  lr: {config['lr']:.6f}")
            print(f"  Objectives:")
            for j, name in enumerate(self.objective_names):
                print(f"    {name}: {self.objectives_display[i, j]:.4f}")

    def generate_report(self, output_dir: str = "artifacts/pareto/report"):
        """
        Generate complete visualization report.

        Args:
            output_dir: Output directory for report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Generating Pareto front report in: {output_path}")
        print()

        # Print summaries
        self.print_summary()
        self.print_configurations()

        if not HAS_MATPLOTLIB:
            print("\nMatplotlib not available. Skipping plots.")
            return

        # Generate plots
        print("\nGenerating visualizations...")

        # 2D plots for key objective pairs
        self.plot_2d_front(0, 1, output_path / "accuracy_vs_latency.png")
        self.plot_2d_front(0, 2, output_path / "accuracy_vs_epr_cv.png")
        self.plot_2d_front(1, 4, output_path / "latency_vs_energy.png")

        # Parallel coordinates
        self.plot_parallel_coordinates(output_path / "parallel_coordinates.png")

        # All pairs
        self.plot_all_pairs(output_path / "pairwise_objectives.png")

        print(f"\nReport generated successfully in: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Pareto front")
    parser.add_argument("--results", type=str, default="artifacts/pareto/pareto_front.json",
                        help="Path to pareto_front.json")
    parser.add_argument("--output", type=str, default="artifacts/pareto/report",
                        help="Output directory for visualizations")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"],
                        help="Output format for plots")

    args = parser.parse_args()

    try:
        visualizer = ParetoVisualizer(args.results)
        visualizer.generate_report(args.output)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
