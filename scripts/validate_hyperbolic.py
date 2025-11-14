"""
Validate hyperbolic geometry integration with NDCG@K metrics.
Gate: NDCG@K +5% vs Euclidean baseline on hierarchical datasets.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import argparse
from typing import Dict, Tuple

from tfan.ctd import PoincareEmbedding, HyperbolicLayer, CTDGate
from tfan.datasets import KnowledgeGraphDataset


def dcg_at_k(scores: np.ndarray, k: int) -> float:
    """Compute Discounted Cumulative Gain at K."""
    scores_k = scores[:k]
    gains = np.power(2, scores_k) - 1
    discounts = np.log2(np.arange(2, k + 2))
    return np.sum(gains / discounts)


def ndcg_at_k(predicted_scores: np.ndarray, true_scores: np.ndarray, k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at K.

    Args:
        predicted_scores: Predicted relevance scores
        true_scores: Ground truth relevance scores
        k: Cutoff position

    Returns:
        NDCG@K value between 0 and 1
    """
    # Sort by predicted scores
    pred_order = np.argsort(-predicted_scores)
    true_at_pred = true_scores[pred_order]

    # DCG with predicted order
    dcg = dcg_at_k(true_at_pred, k)

    # Ideal DCG (best possible ordering)
    ideal_order = np.argsort(-true_scores)
    true_at_ideal = true_scores[ideal_order]
    idcg = dcg_at_k(true_at_ideal, k)

    # NDCG
    if idcg == 0:
        return 0.0

    return dcg / idcg


class EuclideanModel(nn.Module):
    """Baseline Euclidean embedding model."""
    def __init__(self, num_entities: int, embed_dim: int = 128):
        super().__init__()
        self.embeddings = nn.Embedding(num_entities, embed_dim)
        self.fc = nn.Linear(embed_dim * 2, 1)

    def forward(self, head_idx: torch.Tensor, tail_idx: torch.Tensor) -> torch.Tensor:
        """Compute link prediction scores."""
        head_emb = self.embeddings(head_idx)  # [B, D]
        tail_emb = self.embeddings(tail_idx)  # [B, D]

        # Concatenate and score
        combined = torch.cat([head_emb, tail_emb], dim=-1)  # [B, 2D]
        score = self.fc(combined).squeeze(-1)  # [B]

        return score


class HyperbolicModel(nn.Module):
    """Hyperbolic embedding model with Poincaré ball."""
    def __init__(self, num_entities: int, embed_dim: int = 128, c: float = 1.0):
        super().__init__()
        self.embeddings = PoincareEmbedding(num_entities, embed_dim, c=c)
        self.hyperbolic_layer = HyperbolicLayer(embed_dim * 2, 1, c=c)

    def forward(self, head_idx: torch.Tensor, tail_idx: torch.Tensor) -> torch.Tensor:
        """Compute link prediction scores in hyperbolic space."""
        head_emb = self.embeddings(head_idx)  # [B, D]
        tail_emb = self.embeddings(tail_idx)  # [B, D]

        # Concatenate in hyperbolic space
        combined = torch.cat([head_emb, tail_emb], dim=-1)  # [B, 2D]

        # Score with hyperbolic layer
        score = self.hyperbolic_layer(combined).squeeze(-1)  # [B]

        return score


def train_model(model: nn.Module, dataset: KnowledgeGraphDataset,
                num_epochs: int = 50, lr: float = 1e-3) -> nn.Module:
    """Train link prediction model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        # Simple batching
        batch_size = 128
        num_samples = len(dataset)

        for i in range(0, num_samples, batch_size):
            batch_indices = list(range(i, min(i + batch_size, num_samples)))
            if len(batch_indices) == 0:
                continue

            # Get batch data
            batch_data = [dataset[idx] for idx in batch_indices]

            head_idx = torch.tensor([d["head"] for d in batch_data], device=device)
            tail_idx = torch.tensor([d["tail"] for d in batch_data], device=device)
            labels = torch.tensor([d["label"] for d in batch_data], dtype=torch.float32, device=device)

            # Forward pass
            scores = model(head_idx, tail_idx)
            loss = criterion(scores, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return model


def evaluate_ndcg(model: nn.Module, dataset: KnowledgeGraphDataset,
                  k_values: list = [1, 3, 5, 10]) -> Dict[int, float]:
    """
    Evaluate model with NDCG@K metrics.

    Args:
        model: Trained model
        dataset: Evaluation dataset
        k_values: List of K values for NDCG@K

    Returns:
        Dict mapping K to NDCG@K score
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Collect predictions and ground truth
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            data = dataset[idx]
            head_idx = torch.tensor([data["head"]], device=device)
            tail_idx = torch.tensor([data["tail"]], device=device)
            label = data["label"]

            score = model(head_idx, tail_idx).item()

            all_scores.append(score)
            all_labels.append(label)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Compute NDCG@K for each K
    ndcg_results = {}
    for k in k_values:
        if k > len(all_scores):
            k = len(all_scores)

        ndcg = ndcg_at_k(all_scores, all_labels, k)
        ndcg_results[k] = ndcg

    return ndcg_results


def validate_hyperbolic_geometry(dataset_name: str = "FB15k-237",
                                 embed_dim: int = 128,
                                 num_epochs: int = 50) -> Dict:
    """
    Validate hyperbolic geometry with NDCG@K gate.

    Returns:
        Validation results with gate status
    """
    print(f"=== Validating Hyperbolic Geometry on {dataset_name} ===\n")

    # Create dataset
    if dataset_name == "FB15k-237":
        num_entities = 14541
    elif dataset_name == "WordNet":
        num_entities = 40943
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset = KnowledgeGraphDataset(dataset_name=dataset_name, num_entities=num_entities)

    # Train Euclidean baseline
    print("Training Euclidean baseline...")
    euclidean_model = EuclideanModel(num_entities=num_entities, embed_dim=embed_dim)
    euclidean_model = train_model(euclidean_model, dataset, num_epochs=num_epochs)

    # Evaluate Euclidean
    print("\nEvaluating Euclidean model...")
    euclidean_ndcg = evaluate_ndcg(euclidean_model, dataset, k_values=[1, 3, 5, 10])

    # Train Hyperbolic model
    print("\nTraining Hyperbolic model...")
    hyperbolic_model = HyperbolicModel(num_entities=num_entities, embed_dim=embed_dim, c=1.0)
    hyperbolic_model = train_model(hyperbolic_model, dataset, num_epochs=num_epochs)

    # Evaluate Hyperbolic
    print("\nEvaluating Hyperbolic model...")
    hyperbolic_ndcg = evaluate_ndcg(hyperbolic_model, dataset, k_values=[1, 3, 5, 10])

    # Compare results
    print("\n=== NDCG@K Comparison ===")
    print(f"{'K':<5} {'Euclidean':<12} {'Hyperbolic':<12} {'Improvement':<12} {'Gate':<6}")
    print("-" * 55)

    improvements = {}
    gate_passed = True

    for k in [1, 3, 5, 10]:
        euc_score = euclidean_ndcg[k]
        hyp_score = hyperbolic_ndcg[k]
        improvement = ((hyp_score - euc_score) / euc_score) * 100 if euc_score > 0 else 0.0

        improvements[k] = improvement

        gate_status = "✅" if improvement >= 5.0 else "❌"
        if improvement < 5.0:
            gate_passed = False

        print(f"{k:<5} {euc_score:<12.4f} {hyp_score:<12.4f} {improvement:>+10.2f}% {gate_status:<6}")

    # Summary
    print("\n=== Summary ===")
    avg_improvement = sum(improvements.values()) / len(improvements)
    print(f"Average NDCG improvement: {avg_improvement:+.2f}%")
    print(f"Gate (≥+5%): {'✅ PASSED' if gate_passed else '❌ FAILED'}")

    results = {
        "dataset": dataset_name,
        "euclidean_ndcg": euclidean_ndcg,
        "hyperbolic_ndcg": hyperbolic_ndcg,
        "improvements": improvements,
        "avg_improvement": avg_improvement,
        "gate_passed": gate_passed
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate hyperbolic geometry integration")
    parser.add_argument("--dataset", type=str, default="FB15k-237",
                       choices=["FB15k-237", "WordNet"],
                       help="Dataset to use for validation")
    parser.add_argument("--embed-dim", type=int, default=128,
                       help="Embedding dimension")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--output", type=str, default="artifacts/hyperbolic_validation.json",
                       help="Output JSON file")

    args = parser.parse_args()

    # Run validation
    results = validate_hyperbolic_geometry(
        dataset_name=args.dataset,
        embed_dim=args.embed_dim,
        num_epochs=args.epochs
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Exit with appropriate code
    return 0 if results["gate_passed"] else 1


if __name__ == "__main__":
    exit(main())
