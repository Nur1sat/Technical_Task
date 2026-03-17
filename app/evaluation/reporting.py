from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from app.inference.predictor import PredictionResult
from app.labels import normalize_label


def infer_expected_label_from_filename(path: str | Path) -> str | None:
    stem = Path(path).stem
    prefix = stem.split("__", 1)[0]

    try:
        return normalize_label(prefix)
    except ValueError:
        return None


def summarize_external_predictions(predictions: Iterable[PredictionResult]) -> dict[str, float | int]:
    items = list(predictions)
    labeled = [
        (infer_expected_label_from_filename(item.video_path), item.predicted_label)
        for item in items
    ]
    comparable = [(expected, predicted) for expected, predicted in labeled if expected is not None]

    if not comparable:
        return {
            "total_videos": len(items),
            "labeled_videos": 0,
            "accuracy": 0.0,
        }

    correct = sum(1 for expected, predicted in comparable if expected == predicted)
    return {
        "total_videos": len(items),
        "labeled_videos": len(comparable),
        "accuracy": correct / len(comparable),
    }


def write_predictions_json(path: str | Path, predictions: Iterable[PredictionResult]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = [prediction.to_dict() for prediction in predictions]
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_predictions_csv(path: str | Path, predictions: Iterable[PredictionResult]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = list(predictions)

    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["video_path", "predicted_label", "confidence", "probabilities"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "video_path": row.video_path,
                    "predicted_label": row.predicted_label,
                    "confidence": row.confidence,
                    "probabilities": json.dumps(row.probabilities),
                }
            )
