"""Tools for consolidating adapter weights with low-rank SVD."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor


def svd_consolidate(
    weight: Tensor, rank: int = 64, return_factors: bool = False
) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]]:
    """Compute a rank-constrained approximation of ``weight`` via SVD."""

    if weight.ndim != 2:
        raise ValueError("svd_consolidate expects a 2D weight matrix")
    if rank <= 0:
        raise ValueError("rank must be positive")

    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    max_rank = min(rank, s.shape[-1])
    u_r = u[:, :max_rank]
    s_r = s[:max_rank]
    vh_r = vh[:max_rank, :]
    approx = (u_r * s_r) @ vh_r
    if return_factors:
        return approx, (u_r, s_r, vh_r)
    return approx


def _load_tensor(path: Path) -> Tensor:
    if path.suffix in {".pt", ".pth"}:
        data = torch.load(path, map_location="cpu")
        if isinstance(data, Tensor):
            return data
        raise ValueError("Loaded object is not a Tensor")
    if path.suffix == ".npy":
        return torch.from_numpy(np.load(path))
    raise ValueError(f"Unsupported file extension: {path.suffix}")


def _save_tensor(path: Path, tensor: Tensor) -> None:
    if path.suffix in {".pt", ".pth"}:
        torch.save(tensor, path)
    elif path.suffix == ".npy":
        np.save(path, tensor.cpu().numpy())
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Consolidate adapter weights using rank-limited SVD")
    parser.add_argument("input", type=Path, help="Path to the weight tensor (.pt/.pth/.npy)")
    parser.add_argument("output", type=Path, help="Where to save the consolidated tensor")
    parser.add_argument("--rank", type=int, default=64, help="Target rank (default: 64)")
    parser.add_argument("--return-factors", action="store_true", help="Also persist factor matrices")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    weight = _load_tensor(args.input)
    result = svd_consolidate(weight, rank=args.rank, return_factors=args.return_factors)
    if args.return_factors:
        consolidated, factors = result  # type: ignore[assignment]
    else:
        consolidated = result  # type: ignore[assignment]
        factors = None

    _save_tensor(args.output, consolidated)

    if factors is not None:
        base = args.output.with_suffix("")
        u, s, vh = factors
        _save_tensor(base.with_name(base.name + "_U.pt"), u)
        _save_tensor(base.with_name(base.name + "_S.pt"), s)
        _save_tensor(base.with_name(base.name + "_Vh.pt"), vh)


if __name__ == "__main__":
    main()
