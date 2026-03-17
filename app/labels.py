from __future__ import annotations

from pathlib import Path
import re

CLASS_NAMES = ("nonviolence", "violence")
LABEL_TO_INDEX = {label: index for index, label in enumerate(CLASS_NAMES)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"}

VIOLENCE_ALIASES = {
    "violence",
    "violent",
    "fight",
    "fighting",
}
NON_VIOLENCE_ALIASES = {
    "nonviolence",
    "nonviolent",
    "nonviolents",
    "nonviolentvideos",
    "normal",
    "safe",
}


def normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def normalize_label(value: str) -> str:
    token = normalize_token(value)

    if token in VIOLENCE_ALIASES:
        return "violence"
    if token in NON_VIOLENCE_ALIASES:
        return "nonviolence"

    raise ValueError(f"Cannot normalize label: {value}")


def infer_label_from_path(path: str | Path) -> str | None:
    candidate = Path(path)
    tokens: list[str] = []

    tokens.extend(filter(None, re.split(r"[_\-\s]+", candidate.stem)))
    for part in reversed(candidate.parts):
        tokens.extend(filter(None, re.split(r"[_\-\s]+", part)))

    for token in tokens:
        normalized = normalize_token(token)
        if normalized in NON_VIOLENCE_ALIASES:
            return "nonviolence"
        if normalized in VIOLENCE_ALIASES:
            return "violence"

    return None
