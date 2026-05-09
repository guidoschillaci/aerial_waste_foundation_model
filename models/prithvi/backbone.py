"""
Prithvi-EO-2.0 backbone with optional LoRA fine-tuning via PEFT.

Supports:
  - Binary classification head (waste / no-waste)
  - Multi-label head (15 waste types)
  - Pluggable decoder (UperNet, UNet, identity) for segmentation

Architecture notes:
  - Prithvi-EO-2.0-300M is a ViT-L with 3D patch embeddings
  - Embed dim: 1024, depth: 24 layers
  - Pretrained on 4.2M global HLS (Harmonized Landsat-Sentinel-2) samples
  - Accepts (B, C, T, H, W) — we use T=1 for AerialWaste single-timestamp
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("[warn] peft not installed. LoRA fine-tuning disabled. pip install peft")

try:
    from terratorch.registry import BACKBONE_REGISTRY
    TERRATORCH_AVAILABLE = True
except ImportError:
    TERRATORCH_AVAILABLE = False
    print("[warn] terratorch not installed. pip install terratorch")


@dataclass
class LoRAConfig:
    """LoRA hyperparameters, matching IBM peft-geofm paper defaults."""
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    bias: str = "none"
    # Modules to inject LoRA into (Prithvi-EO-2.0 ViT layer names)
    target_modules: list[str] = None

    def __post_init__(self):
        if self.target_modules is None:
            # Q, V projections + MLP layers — matches IBM peft-geofm repo
            self.target_modules = [
                "qkv.q_linear",
                "qkv.v_linear",
                "mlp.fc1",
                "mlp.fc2",
            ]


class PrithviClassifier(nn.Module):
    """
    Prithvi-EO-2.0 backbone + classification head for AerialWaste.

    Can be used with:
      - Frozen backbone (linear probe)
      - LoRA fine-tuning (~1.6% trainable params)
      - Full fine-tuning (all params)
    """

    EMBED_DIM = {
        "prithvi_eo_v2_300": 1024,
        "prithvi_eo_v2_600": 1280,
        "prithvi_eo_v2_tiny": 512,
    }

    def __init__(
        self,
        backbone_name: str = "prithvi_eo_v2_300",
        num_classes: int = 2,
        task: str = "classification",        # "classification" | "multilabel"
        freeze_backbone: bool = False,
        lora_config: Optional[LoRAConfig] = None,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()
        self.task = task
        self.backbone_name = backbone_name

        # ── Load Prithvi backbone ─────────────────────────────────────────────
        if not TERRATORCH_AVAILABLE:
            raise ImportError("terratorch is required. pip install terratorch")

        print(f"Loading {backbone_name} (pretrained={pretrained}) ...")
        self.backbone = BACKBONE_REGISTRY.build(
            backbone_name,
            pretrained=pretrained,
            # RGB only — TerraTorch adapts patch embeddings automatically
            bands=["RED", "GREEN", "BLUE"],
            num_frames=1,   # single timestamp for AerialWaste
        )

        embed_dim = self.EMBED_DIM.get(backbone_name, 1024)

        # ── Apply LoRA (if requested) ─────────────────────────────────────────
        if lora_config is not None:
            if not PEFT_AVAILABLE:
                raise ImportError("peft is required for LoRA. pip install peft")

            peft_config = LoraConfig(
                task_type=None,  # FEATURE_EXTRACTION injects input_ids; None uses plain LoraModel
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                target_modules=lora_config.target_modules,
                lora_dropout=lora_config.lora_dropout,
                bias=lora_config.bias,
            )
            self.backbone = get_peft_model(self.backbone, peft_config)
            self.backbone.print_trainable_parameters()

        elif freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen (linear probe mode)")

        # ── Classification head ───────────────────────────────────────────────
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) — RGB images, normalized

        Returns:
            logits: (B, num_classes)
        """
        # Prithvi expects (B, C, T, H, W) — add time dimension
        if x.dim() == 4:
            x = x.unsqueeze(2)  # → (B, C, 1, H, W)

        features = self.backbone(x)

        # features is a list of feature maps from different ViT layers
        # Take the last one: (B, embed_dim, H', W')
        feat = features[-1] if isinstance(features, (list, tuple)) else features

        # If still spatial, pool to (B, embed_dim)
        if feat.dim() == 4:
            feat = self.pool(feat).flatten(1)
        elif feat.dim() == 3:
            # (B, N_tokens, embed_dim) → mean pool over tokens
            feat = feat.mean(dim=1)

        return self.head(feat)

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
            "trainable_pct": 100.0 * trainable / total,
        }


def build_prithvi_lora(
    num_classes: int = 2,
    task: str = "classification",
    lora_r: int = 16,
    lora_alpha: int = 16,
    backbone: str = "prithvi_eo_v2_300",
) -> PrithviClassifier:
    """Convenience builder matching IBM peft-geofm paper defaults."""
    lora_config = LoRAConfig(r=lora_r, lora_alpha=lora_alpha)
    return PrithviClassifier(
        backbone_name=backbone,
        num_classes=num_classes,
        task=task,
        lora_config=lora_config,
    )
