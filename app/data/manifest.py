from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path
import random

from app.labels import LABEL_TO_INDEX, VIDEO_EXTENSIONS, infer_label_from_path, normalize_token

SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "val",
}


@dataclass(slots=True)
class ManifestRecord:
    video_path: str
    label: str
    split: str

    @property
    def label_index(self) -> int:
        return LABEL_TO_INDEX[self.label]


def infer_split_from_path(path: Path) -> str | None:
    for part in path.parts:
        normalized = normalize_token(part)
        if normalized in SPLIT_ALIASES:
            return SPLIT_ALIASES[normalized]
    return None


def scan_dataset(data_dir: str | Path) -> list[ManifestRecord]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    records: list[ManifestRecord] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        label = infer_label_from_path(path)
        if label is None:
            continue

        split = infer_split_from_path(path) or "unspecified"
        records.append(
            ManifestRecord(
                video_path=str(path.resolve()),
                label=label,
                split=split,
            )
        )

    if not records:
        raise RuntimeError(f"No labeled videos found inside {root}")

    return records


def split_records(
    records: list[ManifestRecord],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[ManifestRecord], list[ManifestRecord]]:
    by_label: dict[str, list[ManifestRecord]] = {}
    for record in records:
        by_label.setdefault(record.label, []).append(record)

    rng = random.Random(seed)
    train_records: list[ManifestRecord] = []
    val_records: list[ManifestRecord] = []

    for label, label_records in by_label.items():
        bucket = label_records[:]
        rng.shuffle(bucket)
        val_size = max(1, int(round(len(bucket) * val_ratio)))

        for item in bucket[:val_size]:
            val_records.append(ManifestRecord(item.video_path, label, "val"))
        for item in bucket[val_size:]:
            train_records.append(ManifestRecord(item.video_path, label, "train"))

    train_records.sort(key=lambda item: item.video_path)
    val_records.sort(key=lambda item: item.video_path)
    return train_records, val_records


def build_manifests(
    records: list[ManifestRecord],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[ManifestRecord], list[ManifestRecord]]:
    train_records = [record for record in records if record.split == "train"]
    val_records = [record for record in records if record.split == "val"]

    if train_records and val_records:
        return train_records, val_records

    normalized = [
        ManifestRecord(record.video_path, record.label, "unspecified")
        for record in records
    ]
    return split_records(normalized, val_ratio=val_ratio, seed=seed)


def write_manifest(path: str | Path, records: list[ManifestRecord]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_path", "label", "split"])
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def read_manifest(path: str | Path) -> list[ManifestRecord]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            ManifestRecord(
                video_path=row["video_path"],
                label=row["label"],
                split=row.get("split", "unknown"),
            )
            for row in reader
        ]
