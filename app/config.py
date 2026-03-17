from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrainConfig:
    data_dir: str = "data/raw"
    train_manifest: str = "data/manifests/train.csv"
    val_manifest: str = "data/manifests/val.csv"
    output_dir: str = "artifacts/run_r3d18"
    backbone: str = "r3d_18"
    pretrained: bool = True
    dropout: float = 0.25
    batch_size: int = 4
    epochs: int = 12
    freeze_backbone_epochs: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    num_frames: int = 16
    image_size: int = 112
    max_video_frames: int = 160
    seed: int = 42
    amp: bool = True
    device: str = "auto"
    max_grad_norm: float = 1.0

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)

    def to_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def as_metadata(self) -> dict[str, Any]:
        return asdict(self)
