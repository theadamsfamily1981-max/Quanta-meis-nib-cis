"""PyTorch implementation of the Topological Field-Adaptive Network (T-FAN)."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


@dataclass
class TFANConfig:
    input_dim: int = 256
    topo_dim: int = 48
    hidden_dim: int = 512
    latent_dim: int = 256
    output_dim: int = 12
    num_blocks: int = 4
    dropout_rate: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
class FieldKernel(nn.Module):
    def __init__(self, hidden_dim: int, laplacian_dim: int) -> None:
        super().__init__()
        self.project_x = nn.Linear(hidden_dim, hidden_dim)
        self.project_lap = nn.Linear(laplacian_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, laplacian: Tensor) -> Tensor:
        lap_summary = laplacian.mean(dim=-1)
        return self.norm(self.project_x(x) + self.project_lap(lap_summary))


class TopologicalBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float, laplacian_dim: int) -> None:
        super().__init__()
        self.kernel = FieldKernel(hidden_dim, laplacian_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, topo_gate: Tensor, laplacian: Tensor) -> Tensor:
        fused = self.kernel(x, laplacian)
        fused = self.linear(fused * topo_gate)
        fused = F.gelu(fused)
        fused = self.dropout(fused)
        return self.norm(x + fused)


class ReadoutHead(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class TopologicalFieldAdaptiveNetwork(nn.Module):
    def __init__(self, config: TFANConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(config.input_dim, config.hidden_dim)
        self.persistence_encoder = nn.Sequential(
            nn.Linear(config.topo_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                TopologicalBlock(config.hidden_dim, config.dropout_rate, config.input_dim)
                for _ in range(config.num_blocks)
            ]
        )
        self.readout = ReadoutHead(config.hidden_dim, config.output_dim)

    def forward(self, inputs: Tensor, persistence: Tensor, laplacian: Tensor) -> Tensor:
        x = F.gelu(self.encoder(inputs))
        topo_gate = torch.sigmoid(self.persistence_encoder(persistence))
        for block in self.blocks:
            x = block(x, topo_gate, laplacian)
        return self.readout(x)


def compute_laplacian(batch: Tensor) -> Tensor:
    cov = batch.unsqueeze(-1) * batch.unsqueeze(-2)
    norm = cov.norm(dim=(-2, -1), keepdim=True) + 1e-6
    adjacency = cov / norm
    degree = adjacency.sum(dim=-1)
    identity = torch.eye(adjacency.size(-1), device=batch.device)
    degree_matrix = torch.einsum("bi,ij->bij", degree, identity)
    return degree_matrix - adjacency


def generate_synthetic_batch(
    batch_size: int, config: TFANConfig, device: torch.device
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    features = torch.randn(batch_size, config.input_dim, device=device)
    persistence = torch.rand(batch_size, config.topo_dim, device=device)
    laplacian = compute_laplacian(features)
    logits = features[:, : config.output_dim]
    logits = logits + 0.1 * torch.randn_like(logits)
    labels = torch.argmax(logits, dim=-1)
    noise = torch.randint(0, 2, labels.shape, device=device)
    labels = (labels + noise) % config.output_dim
    return features, persistence, laplacian, labels


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    config = TFANConfig(
        input_dim=args.input_dim,
        topo_dim=args.topo_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        output_dim=args.output_dim,
        num_blocks=args.num_blocks,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    model = TopologicalFieldAdaptiveNetwork(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    print(f"Running on {device} with {count_parameters(model):,} parameters")

    for epoch in range(args.epochs):
        model.train()
        features, persistence, laplacian, labels = generate_synthetic_batch(
            args.batch_size, config, device
        )

        optimizer.zero_grad()
        outputs = model(features, persistence, laplacian)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = torch.argmax(outputs, dim=-1)
            accuracy = (preds == labels).float().mean().item()
        print(f"Epoch {epoch + 1:02d} | loss={loss.item():.4f} accuracy={accuracy:.4f}")

    model.eval()
    with torch.no_grad():
        features, persistence, laplacian, labels = generate_synthetic_batch(
            args.batch_size, config, device
        )
        outputs = model(features, persistence, laplacian)
        val_loss = criterion(outputs, labels).item()
        val_acc = (torch.argmax(outputs, dim=-1) == labels).float().mean().item()
    print(f"Validation | loss={val_loss:.4f} accuracy={val_acc:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--input-dim", type=int, default=256)
    parser.add_argument("--output-dim", type=int, default=12)
    parser.add_argument("--topo-dim", type=int, default=48)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    train(parser.parse_args())
