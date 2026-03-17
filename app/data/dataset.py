from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from app.data.manifest import ManifestRecord, read_manifest
from app.data.video_reader import build_clip_tensor, load_video_frames


class ViolenceVideoDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(
        self,
        manifest_path: str | Path,
        num_frames: int,
        image_size: int,
        max_video_frames: int,
        train: bool = False,
    ) -> None:
        self.records: list[ManifestRecord] = read_manifest(manifest_path)
        self.num_frames = num_frames
        self.image_size = image_size
        self.max_video_frames = max_video_frames
        self.train = train

        if not self.records:
            raise RuntimeError(f"Manifest is empty: {manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records[index]
        frames = load_video_frames(record.video_path, max_frames=self.max_video_frames)
        clip = build_clip_tensor(
            frames,
            num_frames=self.num_frames,
            image_size=self.image_size,
            train=self.train,
        )

        video_tensor = torch.from_numpy(clip)
        label_tensor = torch.tensor(record.label_index, dtype=torch.long)
        return video_tensor, label_tensor, record.video_path
