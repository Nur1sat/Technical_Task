from __future__ import annotations

from pathlib import Path
import random

import cv2
import numpy as np

KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
KINETICS_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


def load_video_frames(video_path: str | Path, max_frames: int = 160) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    read_stride = 1
    if total_frames > max_frames > 0:
        read_stride = max(1, total_frames // max_frames)

    frames: list[np.ndarray] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index % read_stride == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
            if len(frames) >= max_frames:
                break

        frame_index += 1

    capture.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")

    return frames


def sample_frame_indices(total_frames: int, num_frames: int, train: bool) -> list[int]:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")

    if total_frames == 1:
        return [0] * num_frames

    if total_frames <= num_frames:
        indices = list(range(total_frames))
        indices.extend([total_frames - 1] * (num_frames - total_frames))
        return indices

    if not train:
        return np.linspace(0, total_frames - 1, num=num_frames, dtype=int).tolist()

    # Add temporal jitter but still cover the whole clip.
    span = total_frames / num_frames
    indices: list[int] = []
    for step in range(num_frames):
        start = int(step * span)
        end = max(start + 1, int((step + 1) * span))
        indices.append(random.randint(start, min(end, total_frames - 1)))
    return indices


def build_clip_tensor(
    frames: list[np.ndarray],
    num_frames: int,
    image_size: int,
    train: bool = False,
) -> np.ndarray:
    indices = sample_frame_indices(len(frames), num_frames=num_frames, train=train)
    clip = [frames[index] for index in indices]

    do_flip = train and random.random() < 0.5
    brightness = random.uniform(0.9, 1.1) if train else 1.0

    processed: list[np.ndarray] = []
    for frame in clip:
        resized = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        if do_flip:
            resized = np.ascontiguousarray(resized[:, ::-1])

        frame_float = resized.astype(np.float32) / 255.0
        if brightness != 1.0:
            frame_float = np.clip(frame_float * brightness, 0.0, 1.0)

        processed.append(frame_float)

    clip_array = np.stack(processed, axis=0)
    clip_array = (clip_array - KINETICS_MEAN) / KINETICS_STD
    clip_array = np.transpose(clip_array, (3, 0, 1, 2))
    return clip_array.astype(np.float32)
