"""Tiny end-to-end demo wiring all TLS components together."""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

    class _DummyNN:  # type: ignore
        class Module:
            pass

        def __getattr__(self, name: str):
            raise RuntimeError("PyTorch is required for the toy training loop")

    nn = _DummyNN()  # type: ignore

from ..components.fdr import WeightFluctuationTracker, control_from_metrics
from ..components.fep import variational_free_energy
from ..policies.scheduler import EarlyStopController, FDRLRScheduler
from ..policies.tls import TLSPolicy, TLSPolicyConfig


@dataclass
class ToyConfig:
    latent_dim: int = 8
    hidden_dim: int = 64
    steps: int = 200
    batch_size: int = 64
    lr: float = 3e-3


class ToyVAE(nn.Module):
    def __init__(self, input_dim: int, config: ToyConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(config.hidden_dim, config.latent_dim)
        self.logvar_head = nn.Linear(config.hidden_dim, config.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        recon = self.decoder(z)
        return recon, mu, logvar


def make_dataset(n: int = 2048) -> np.ndarray:
    rng = np.random.default_rng(0)
    angles = rng.uniform(0, 2 * np.pi, size=n)
    radii = 0.5 + 0.5 * rng.normal(size=n)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    z = radii * np.sin(2 * angles)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def train(args: argparse.Namespace) -> None:
    if torch is None:  # pragma: no cover
        raise ImportError("PyTorch is required for the toy training loop")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = make_dataset()
    config = ToyConfig(steps=args.steps, batch_size=args.batch)
    policy = TLSPolicy(TLSPolicyConfig(embed_dim=config.hidden_dim))
    model = ToyVAE(data.shape[1], config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = FDRLRScheduler(optimizer, base_lr=config.lr)
    early_stop = EarlyStopController(patience=20)
    fluct_tracker = WeightFluctuationTracker()

    tensor_data = torch.tensor(data, device=device)

    for step in range(config.steps):
        idx = torch.randint(0, tensor_data.size(0), (config.batch_size,), device=device)
        batch = tensor_data[idx]
        signature = policy.encode_geometry(batch.detach().cpu().numpy())

        recon, mu, logvar = model(batch)
        loss = variational_free_energy(
            recon,
            batch,
            posterior_mu=mu,
            posterior_logvar=logvar,
            topo_signature=signature,
            topo_weight=1e-2,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        fluctuation = fluct_tracker.update(model.parameters())
        curvature = float(loss.detach().cpu())
        signal = control_from_metrics(
            curvature,
            fluctuation,
            lr_base=config.lr,
            temperature_base=policy.get_temperature(),
            curvature_clip=10.0,
            fluctuation_clip=1.0,
        )
        scheduler.update(signal)

        if early_stop.step(float(loss.detach().cpu())):
            print(f"Early stop triggered at step {step}")
            break

        if step % 20 == 0:
            print(f"Step {step:04d} | Loss: {loss.item():.4f} | Temp: {policy.get_temperature():.3f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
