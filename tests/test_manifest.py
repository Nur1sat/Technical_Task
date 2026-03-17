from pathlib import Path

from app.data.manifest import ManifestRecord, build_manifests, infer_split_from_path
from app.labels import infer_label_from_path


def test_infer_label_from_common_dataset_paths() -> None:
    assert infer_label_from_path(Path("/tmp/Violence/demo.mp4")) == "violence"
    assert infer_label_from_path(Path("/tmp/NonViolence/demo.mp4")) == "nonviolence"


def test_infer_split_from_path_detects_validation_alias() -> None:
    assert infer_split_from_path(Path("/tmp/train/Violence/a.mp4")) == "train"
    assert infer_split_from_path(Path("/tmp/validation/Violence/a.mp4")) == "val"


def test_build_manifests_preserves_explicit_train_val_split() -> None:
    records = [
        ManifestRecord("/tmp/train/Violence/1.mp4", "violence", "train"),
        ManifestRecord("/tmp/train/NonViolence/1.mp4", "nonviolence", "train"),
        ManifestRecord("/tmp/val/Violence/1.mp4", "violence", "val"),
        ManifestRecord("/tmp/val/NonViolence/1.mp4", "nonviolence", "val"),
    ]

    train_records, val_records = build_manifests(records, val_ratio=0.2, seed=7)

    assert len(train_records) == 2
    assert len(val_records) == 2
    assert all(record.split == "train" for record in train_records)
    assert all(record.split == "val" for record in val_records)


def test_build_manifests_falls_back_to_balanced_split() -> None:
    records = [
        ManifestRecord(f"/tmp/Violence/{index}.mp4", "violence", "unspecified")
        for index in range(10)
    ] + [
        ManifestRecord(f"/tmp/NonViolence/{index}.mp4", "nonviolence", "unspecified")
        for index in range(10)
    ]

    train_records, val_records = build_manifests(records, val_ratio=0.2, seed=11)

    assert len(train_records) == 16
    assert len(val_records) == 4
    assert sum(record.label == "violence" for record in val_records) == 2
    assert sum(record.label == "nonviolence" for record in val_records) == 2
