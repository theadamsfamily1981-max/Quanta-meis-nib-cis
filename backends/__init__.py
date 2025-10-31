try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False

__all__ = ["TORCH_AVAILABLE"]
