"""
ResNet50 + FPN baseline — reproduction of the original AerialWaste paper model.

Torres & Fraternali (Scientific Data, 2023):
  "The architecture of the binary classifier extending Resnet50 with FPN links
   used for the technical validation of the AerialWaste data set."

Reported results:
  - All sources: 87.9% Average Precision
  - AGEA only:   94.5% Average Precision
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.ops import FeaturePyramidNetwork


class ResNet50FPN(nn.Module):
    """
    ResNet50 backbone with FPN neck and binary classification head.

    Reproduces the architecture from Torres & Fraternali (2023) as closely
    as possible from the paper description.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        fpn_out_channels: int = 256,
        dropout: float = 0.1,
        freeze_early_layers: bool = True,
    ):
        super().__init__()

        # ── ResNet50 backbone (ImageNet pretrained) ───────────────────────────
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = tv_models.resnet50(weights=weights)

        # Extract intermediate feature maps (C2–C5) for FPN
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1   # stride 4,  256 channels
        self.layer2 = resnet.layer2   # stride 8,  512 channels
        self.layer3 = resnet.layer3   # stride 16, 1024 channels
        self.layer4 = resnet.layer4   # stride 32, 2048 channels

        # Freeze first two layers as in Torres & Fraternali (2023)
        if freeze_early_layers:
            for param in list(self.layer0.parameters()) + list(self.layer1.parameters()):
                param.requires_grad = False

        # ── Feature Pyramid Network ───────────────────────────────────────────
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=fpn_out_channels,
        )

        # ── Classification head ───────────────────────────────────────────────
        # Global average pool over the highest-resolution FPN level
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(fpn_out_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            logits: (B, num_classes)
        """
        c0 = self.layer0(x)
        c2 = self.layer1(c0)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        fpn_features = self.fpn({
            "0": c2,
            "1": c3,
            "2": c4,
            "3": c5,
        })

        # Use highest-resolution FPN output (P2)
        p2 = fpn_features["0"]
        pooled = self.pool(p2)
        return self.classifier(pooled)

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "trainable_pct": 100.0 * trainable / total,
        }
