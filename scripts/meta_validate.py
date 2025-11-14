"""
Meta-learning validation script for TFAN.
Validates few-shot learning capabilities and adaptation speed.
"""
import torch
import torch.nn as nn
import argparse
from pathlib import Path
import json
import time

from tfan.meta_trainer import MAMLTrainer, compute_adaptation_metrics
from tfan.meta_datasets import create_meta_dataloaders
from tfan.io import write_json


class SimpleMLPModel(nn.Module):
    """Simple MLP for meta-learning validation."""
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def validate_meta_learning(
        checkpoint_path: Optional[str] = None,
        num_shots: int = 5,
        num_queries: int = 15,
        num_val_tasks: int = 100,
        output_path: str = "artifacts/meta_validation.json") -> Dict:
    """
    Validate meta-learned model on few-shot tasks.

    Returns:
        Validation results dict
    """
    print("=== MAML Meta-Learning Validation ===\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    model = SimpleMLPModel(input_dim=1, hidden_dim=64, output_dim=1).to(device)

    # Create meta-trainer
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    maml_trainer = MAMLTrainer(
        model=model,
        meta_optimizer=meta_optimizer,
        inner_lr=0.01,
        num_inner_steps=5,
        first_order=True,  # Use FOMAML for speed
        use_fdt_meta=False,  # Disable for validation
        device=device
    )

    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        maml_trainer.load_meta_checkpoint(checkpoint_path)
        print(f"Loaded meta-checkpoint from: {checkpoint_path}\n")

    # Create validation tasks
    _, meta_val_loader = create_meta_dataloaders(
        dataset_type="sinusoid",
        num_train_tasks=0,
        num_val_tasks=num_val_tasks,
        num_shots=num_shots,
        num_queries=num_queries,
        tasks_per_batch=1
    )

    criterion = nn.MSELoss()

    # Validate
    print(f"Validating on {num_val_tasks} tasks...")
    print(f"K-shot: {num_shots}, Queries: {num_queries}\n")

    val_stats = maml_trainer.meta_validate(
        list(meta_val_loader),
        criterion
    )

    # Few-shot accuracy gate
    baseline_loss = 2.0  # Expected loss without adaptation
    improvement = (baseline_loss - val_stats['val_loss']) / baseline_loss * 100

    print("\n=== Validation Results ===")
    print(f"Validation loss: {val_stats['val_loss']:.4f}")
    print(f"Val loss std: {val_stats['val_loss_std']:.4f}")
    print(f"Improvement vs baseline: {improvement:.1f}%")
    print(f"Avg adaptation time: {val_stats['avg_adaptation_time']*1000:.1f}ms")

    # Gates
    few_shot_gate = improvement >= 60  # ≥60% improvement
    adaptation_gate = val_stats['avg_adaptation_time'] < 0.1  # <100ms

    print("\n=== Gates ===")
    print(f"Few-shot accuracy (≥60% improvement): {'✅' if few_shot_gate else '❌'} ({improvement:.1f}%)")
    print(f"Adaptation speed (<100ms): {'✅' if adaptation_gate else '❌'} ({val_stats['avg_adaptation_time']*1000:.1f}ms)")

    # Save results
    results = {
        "val_stats": val_stats,
        "improvement_percent": improvement,
        "gates": {
            "few_shot_accuracy": few_shot_gate,
            "adaptation_speed": adaptation_gate,
            "all_passed": few_shot_gate and adaptation_gate
        },
        "config": {
            "num_shots": num_shots,
            "num_queries": num_queries,
            "num_val_tasks": num_val_tasks
        }
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, results)

    print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate MAML meta-learning")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to meta-learned checkpoint")
    parser.add_argument("--num-shots", type=int, default=5,
                       help="Number of support examples (K-shot)")
    parser.add_argument("--num-queries", type=int, default=15,
                       help="Number of query examples")
    parser.add_argument("--num-val-tasks", type=int, default=100,
                       help="Number of validation tasks")
    parser.add_argument("--output", type=str,
                       default="artifacts/meta_validation.json",
                       help="Output JSON file")

    args = parser.parse_args()

    results = validate_meta_learning(
        checkpoint_path=args.checkpoint,
        num_shots=args.num_shots,
        num_queries=args.num_queries,
        num_val_tasks=args.num_val_tasks,
        output_path=args.output
    )

    # Exit with appropriate code
    return 0 if results["gates"]["all_passed"] else 1


if __name__ == "__main__":
    exit(main())
