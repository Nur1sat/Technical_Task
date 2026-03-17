from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.config import TrainConfig
from app.models.video_classifier import ViolenceVideoClassifier, freeze_backbone, unfreeze_backbone
from app.training.checkpoints import save_model_bundle
from app.training.metrics import MetricSummary, compute_classification_metrics


def run_epoch(
    model: ViolenceVideoClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: Optimizer | None = None,
    max_grad_norm: float | None = None,
    use_amp: bool = False,
) -> MetricSummary:
    is_train = optimizer is not None
    model.train(is_train)

    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device == "cuda")
    progress = tqdm(loader, leave=False)

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for videos, labels, _paths in progress:
            videos = videos.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            autocast_enabled = use_amp and device == "cuda"
            with torch.amp.autocast(device_type="cuda", enabled=autocast_enabled):
                logits = model(videos)
                loss = criterion(logits, labels)

            if is_train:
                scaler.scale(loss).backward()
                if max_grad_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

            predictions = torch.argmax(logits, dim=1)
            losses.append(float(loss.detach().cpu().item()))
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(predictions.detach().cpu().tolist())

            progress.set_postfix(loss=f"{losses[-1]:.4f}")

    return compute_classification_metrics(losses=losses, y_true=y_true, y_pred=y_pred)


def fit(
    model: ViolenceVideoClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    config: TrainConfig,
    device: str,
    class_names: list[str],
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    best_accuracy = -1.0
    best_summary: dict[str, Any] = {}

    if config.freeze_backbone_epochs > 0:
        freeze_backbone(model)

    for epoch in range(1, config.epochs + 1):
        if config.freeze_backbone_epochs and epoch == config.freeze_backbone_epochs + 1:
            unfreeze_backbone(model)

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            max_grad_norm=config.max_grad_norm,
            use_amp=config.amp,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            use_amp=False,
        )

        if scheduler is not None:
            scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics.to_dict(),
            "val": val_metrics.to_dict(),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics.loss:.4f} "
            f"val_loss={val_metrics.loss:.4f} "
            f"val_acc={val_metrics.accuracy:.4f} "
            f"val_f1={val_metrics.f1:.4f}"
        )

        if val_metrics.accuracy >= best_accuracy:
            best_accuracy = val_metrics.accuracy
            metadata = {
                "class_names": class_names,
                "backbone": model.backbone_name,
                "dropout": config.dropout,
                "num_frames": config.num_frames,
                "image_size": config.image_size,
                "max_video_frames": config.max_video_frames,
                "config": config.as_metadata(),
            }
            best_model_path = output_dir / "best_model.safetensors"

            # Keep the runtime metadata next to weights in the same file.
            save_model_bundle(best_model_path, model.state_dict(), metadata)
            best_summary = {
                "model_path": str(best_model_path),
                "metrics": val_metrics.to_dict(),
            }

        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

    summary = {
        "best": best_summary,
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
