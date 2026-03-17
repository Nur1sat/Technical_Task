from __future__ import annotations


def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested

    import torch

    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and mps_backend.is_available():
        return "mps"

    return "cpu"
