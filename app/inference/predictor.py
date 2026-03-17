from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from app.data.video_reader import build_clip_tensor, load_video_frames
from app.labels import INDEX_TO_LABEL
from app.models.video_classifier import ViolenceVideoClassifier
from app.training.checkpoints import load_model_bundle
from app.utils.device import resolve_device


@dataclass(slots=True)
class PredictionResult:
    video_path: str
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]

    def to_dict(self) -> dict[str, str | float | dict[str, float]]:
        return asdict(self)


class Predictor:
    def __init__(self, model_path: str | Path, device: str = "auto") -> None:
        state_dict, metadata = load_model_bundle(model_path)
        self.metadata = metadata
        self.device = resolve_device(device)

        class_names = metadata.get("class_names", ["nonviolence", "violence"])
        self.class_names = list(class_names)

        model = ViolenceVideoClassifier(
            backbone=metadata.get("backbone", "r3d_18"),
            num_classes=len(self.class_names),
            pretrained=False,
            dropout=float(metadata.get("dropout", 0.25)),
        )
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model

        self.num_frames = int(metadata.get("num_frames", 16))
        self.image_size = int(metadata.get("image_size", 112))
        self.max_video_frames = int(metadata.get("max_video_frames", 160))

    @torch.no_grad()
    def predict_video(self, video_path: str | Path) -> PredictionResult:
        frames = load_video_frames(video_path, max_frames=self.max_video_frames)
        clip = build_clip_tensor(
            frames,
            num_frames=self.num_frames,
            image_size=self.image_size,
            train=False,
        )
        tensor = torch.from_numpy(clip).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().tolist()
        best_index = int(torch.argmax(logits, dim=1).item())

        mapped = {
            self.class_names[index] if index < len(self.class_names) else INDEX_TO_LABEL[index]: float(probability)
            for index, probability in enumerate(probabilities)
        }

        return PredictionResult(
            video_path=str(Path(video_path)),
            predicted_label=self.class_names[best_index],
            confidence=float(probabilities[best_index]),
            probabilities=mapped,
        )
