from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from app.config import TrainConfig
from app.data.dataset import ViolenceVideoDataset
from app.labels import CLASS_NAMES
from app.models.video_classifier import ViolenceVideoClassifier
from app.training.engine import fit
from app.utils.device import resolve_device
from app.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train violence video classifier")
    parser.add_argument("--config", default="configs/baseline.json", help="JSON config path")
    parser.add_argument("--output-dir", default=None, help="Optional override for output directory")
    parser.add_argument("--device", default=None, help="Optional override for device: auto/cuda/mps/cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig.from_json(args.config)

    if args.output_dir:
        config.output_dir = args.output_dir
    if args.device:
        config.device = args.device

    set_seed(config.seed)
    device = resolve_device(config.device)
    print(f"device={device}")

    train_manifest = Path(config.train_manifest)
    val_manifest = Path(config.val_manifest)
    if not train_manifest.exists() or not val_manifest.exists():
        raise FileNotFoundError(
            "Train/val manifests are missing. Run `python3 scripts/prepare_dataset.py` first."
        )

    train_dataset = ViolenceVideoDataset(
        manifest_path=train_manifest,
        num_frames=config.num_frames,
        image_size=config.image_size,
        max_video_frames=config.max_video_frames,
        train=True,
    )
    val_dataset = ViolenceVideoDataset(
        manifest_path=val_manifest,
        num_frames=config.num_frames,
        image_size=config.image_size,
        max_video_frames=config.max_video_frames,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device == "cuda",
    )

    model = ViolenceVideoClassifier(
        backbone=config.backbone,
        num_classes=len(CLASS_NAMES),
        pretrained=config.pretrained,
        dropout=config.dropout,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
    criterion = nn.CrossEntropyLoss()

    config_path = Path(config.output_dir) / "used_config.json"
    config.to_json(config_path)

    summary = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        class_names=list(CLASS_NAMES),
    )

    print("best_model=", summary["best"].get("model_path"))
    print("best_val_accuracy=", summary["best"].get("metrics", {}).get("accuracy"))


if __name__ == "__main__":
    main()
