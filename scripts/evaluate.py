from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from torch import nn
from torch.utils.data import DataLoader

from app.data.dataset import ViolenceVideoDataset
from app.labels import CLASS_NAMES
from app.models.video_classifier import ViolenceVideoClassifier
from app.training.checkpoints import load_model_bundle
from app.training.engine import run_epoch
from app.utils.device import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model on validation manifest")
    parser.add_argument("--model", default="artifacts/run_r3d18/best_model.safetensors", help="Path to .safetensors model")
    parser.add_argument("--manifest", default="data/manifests/val.csv", help="Validation manifest path")
    parser.add_argument("--batch-size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--device", default="auto", help="auto/cuda/mps/cpu")
    parser.add_argument("--output", default="artifacts/reports/validation_report.json", help="Where to save metrics JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_dict, metadata = load_model_bundle(args.model)
    device = resolve_device(args.device)

    dataset = ViolenceVideoDataset(
        manifest_path=args.manifest,
        num_frames=int(metadata.get("num_frames", 16)),
        image_size=int(metadata.get("image_size", 112)),
        max_video_frames=int(metadata.get("max_video_frames", 160)),
        train=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )

    model = ViolenceVideoClassifier(
        backbone=metadata.get("backbone", "r3d_18"),
        num_classes=len(metadata.get("class_names", CLASS_NAMES)),
        pretrained=False,
        dropout=float(metadata.get("dropout", 0.25)),
    )
    model.load_state_dict(state_dict)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    metrics = run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        use_amp=False,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()
