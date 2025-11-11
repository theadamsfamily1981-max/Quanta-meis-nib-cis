"""Variational free-energy style loss with a topological prior."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:  # pragma: no cover
    import torch
    from torch import Tensor
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    Tensor = "Tensor"  # type: ignore
    F = None  # type: ignore

from .topology import TopologicalSignature


@dataclass
class FEPLossBreakdown:
    """Expose individual pieces of the FEP objective for debugging."""

    reconstruction: float
    kl: float
    topological_prior: float

    def total(self) -> float:
        return self.reconstruction + self.kl + self.topological_prior


def variational_free_energy(
    recon: Tensor,
    target: Tensor,
    *,
    posterior_mu: Tensor,
    posterior_logvar: Tensor,
    prior_mu: Optional[Tensor] = None,
    prior_logvar: Optional[Tensor] = None,
    topo_signature: Optional[TopologicalSignature] = None,
    topo_weight: float = 1e-2,
    return_breakdown: bool = False,
):
    """Compute the FEP objective used by the project demos."""

    if torch is None:  # pragma: no cover
        raise ImportError("PyTorch is required for variational_free_energy")

    recon_loss = F.mse_loss(recon, target)

    if prior_mu is None:
        prior_mu = torch.zeros_like(posterior_mu)
    if prior_logvar is None:
        prior_logvar = torch.zeros_like(posterior_logvar)

    kl = -0.5 * torch.sum(
        1
        + posterior_logvar
        - prior_logvar
        - ((posterior_mu - prior_mu) ** 2 + posterior_logvar.exp()) / prior_logvar.exp()
    ) / posterior_mu.shape[0]

    topo_penalty = torch.tensor(0.0, device=recon.device)
    if topo_signature is not None and topo_signature.persistence_image is not None:
        image = torch.as_tensor(topo_signature.persistence_image, dtype=recon.dtype, device=recon.device)
        topo_penalty = topo_weight * image.mean()

    total = recon_loss + kl + topo_penalty
    if return_breakdown:
        breakdown = FEPLossBreakdown(
            reconstruction=float(recon_loss.detach().cpu()),
            kl=float(kl.detach().cpu()),
            topological_prior=float(topo_penalty.detach().cpu()),
        )
        return total, breakdown
    return total


__all__ = ["FEPLossBreakdown", "variational_free_energy"]
