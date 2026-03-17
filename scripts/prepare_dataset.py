from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.data.manifest import build_manifests, scan_dataset, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train/val manifests for violence dataset")
    parser.add_argument("--data-dir", default="data/raw", help="Root directory with raw videos")
    parser.add_argument("--manifests-dir", default="data/manifests", help="Where CSV manifests will be saved")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio if dataset has no explicit val split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split reproducibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = scan_dataset(args.data_dir)
    train_records, val_records = build_manifests(records, val_ratio=args.val_ratio, seed=args.seed)

    manifests_dir = Path(args.manifests_dir)
    write_manifest(manifests_dir / "train.csv", train_records)
    write_manifest(manifests_dir / "val.csv", val_records)

    print(f"total_videos={len(records)}")
    print(f"train_videos={len(train_records)}")
    print(f"val_videos={len(val_records)}")
    print(f"manifests_dir={manifests_dir.resolve()}")


if __name__ == "__main__":
    main()
