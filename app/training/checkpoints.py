from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch


def save_model_bundle(
    path: str | Path,
    state_dict: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    cpu_state = {name: tensor.detach().cpu() for name, tensor in state_dict.items()}
    string_metadata = {
        key: value if isinstance(value, str) else json.dumps(value)
        for key, value in metadata.items()
    }

    save_file(cpu_state, str(target), metadata=string_metadata)


def load_model_bundle(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    target = Path(path)
    state_dict = load_file(str(target))

    with safe_open(str(target), framework="pt", device="cpu") as handle:
        raw_metadata = handle.metadata()

    metadata: dict[str, Any] = {}
    for key, value in raw_metadata.items():
        try:
            metadata[key] = json.loads(value)
        except json.JSONDecodeError:
            metadata[key] = value

    return state_dict, metadata
