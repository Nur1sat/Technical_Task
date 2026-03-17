from __future__ import annotations

import torch.nn as nn
from torchvision.models.video import (
    MC3_18_Weights,
    R2Plus1D_18_Weights,
    R3D_18_Weights,
    mc3_18,
    r2plus1d_18,
    r3d_18,
)

BACKBONE_FACTORIES = {
    "r3d_18": (r3d_18, R3D_18_Weights.DEFAULT),
    "mc3_18": (mc3_18, MC3_18_Weights.DEFAULT),
    "r2plus1d_18": (r2plus1d_18, R2Plus1D_18_Weights.DEFAULT),
}


class ViolenceVideoClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "r3d_18",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()

        if backbone not in BACKBONE_FACTORIES:
            raise ValueError(f"Unsupported backbone: {backbone}")

        factory, default_weights = BACKBONE_FACTORIES[backbone]
        weights = default_weights if pretrained else None
        model = factory(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        self.backbone_name = backbone
        self.model = model

    def forward(self, clips):
        return self.model(clips)


def freeze_backbone(model: ViolenceVideoClassifier) -> None:
    for name, parameter in model.model.named_parameters():
        parameter.requires_grad = name.startswith("fc")


def unfreeze_backbone(model: ViolenceVideoClassifier) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True
