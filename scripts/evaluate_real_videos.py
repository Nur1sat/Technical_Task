from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.evaluation.reporting import (
    summarize_external_predictions,
    write_predictions_csv,
    write_predictions_json,
)
from app.inference.predictor import PredictionResult, Predictor
from app.labels import VIDEO_EXTENSIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on external videos")
    parser.add_argument("--model", default="artifacts/run_r3d18/best_model.safetensors", help="Path to .safetensors model")
    parser.add_argument("--input-dir", default="data/external_videos", help="Directory with videos to check")
    parser.add_argument("--output-prefix", default="artifacts/reports/real_videos", help="JSON/CSV output prefix")
    parser.add_argument("--device", default="auto", help="auto/cuda/mps/cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"External videos directory not found: {input_dir}")

    predictor = Predictor(args.model, device=args.device)
    video_paths = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not video_paths:
        raise RuntimeError(f"No videos found inside {input_dir}")

    predictions: list[PredictionResult] = [predictor.predict_video(path) for path in video_paths]
    summary = summarize_external_predictions(predictions)

    output_prefix = Path(args.output_prefix)
    write_predictions_json(output_prefix.with_suffix(".json"), predictions)
    write_predictions_csv(output_prefix.with_suffix(".csv"), predictions)
    output_prefix.with_name(output_prefix.name + "_summary").with_suffix(".json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
