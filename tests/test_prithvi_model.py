"""Tests for Prithvi and baseline model architectures."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestResNet50FPN:
    """Test the ResNet50+FPN baseline model."""

    def test_import(self):
        from models.baseline.resnet50_fpn import ResNet50FPN
        assert ResNet50FPN is not None

    def test_forward_shape(self):
        from models.baseline.resnet50_fpn import ResNet50FPN
        model = ResNet50FPN(num_classes=2, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 2)

    def test_parameter_count(self):
        from models.baseline.resnet50_fpn import ResNet50FPN
        model = ResNet50FPN(num_classes=2, pretrained=False, freeze_early_layers=False)
        params = model.count_parameters()
        assert params["total"] > 1_000_000  # ResNet50 ~25M params
        assert params["trainable"] == params["total"]

    def test_different_image_sizes(self):
        from models.baseline.resnet50_fpn import ResNet50FPN
        model = ResNet50FPN(num_classes=2, pretrained=False)
        model.eval()
        for size in [224, 448]:
            x = torch.randn(1, 3, size, size)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (1, 2), f"Failed for size {size}"


class TestPrithviLoRAConfig:
    """Test LoRA configuration dataclass."""

    def test_default_config(self):
        from models.prithvi.backbone import LoRAConfig
        cfg = LoRAConfig()
        assert cfg.r == 16
        assert cfg.lora_alpha == 16
        assert len(cfg.target_modules) == 4
        assert "qkv.q_linear" in cfg.target_modules
        assert "mlp.fc1" in cfg.target_modules

    def test_custom_config(self):
        from models.prithvi.backbone import LoRAConfig
        cfg = LoRAConfig(r=8, lora_alpha=32)
        assert cfg.r == 8
        assert cfg.lora_alpha == 32

    def test_custom_target_modules(self):
        from models.prithvi.backbone import LoRAConfig
        cfg = LoRAConfig(target_modules=["qkv.q_linear"])
        assert cfg.target_modules == ["qkv.q_linear"]


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("terratorch"),
    reason="terratorch not installed"
)
class TestPrithviClassifier:
    """Test PrithviClassifier (requires terratorch)."""

    def test_build_without_lora(self):
        from models.prithvi.backbone import PrithviClassifier
        model = PrithviClassifier(
            backbone_name="prithvi_eo_v2_300",
            num_classes=2,
            freeze_backbone=True,
            pretrained=False,
        )
        assert model is not None

    def test_build_with_lora(self):
        from models.prithvi.backbone import PrithviClassifier, LoRAConfig
        lora_cfg = LoRAConfig(r=4, lora_alpha=8)
        model = PrithviClassifier(
            backbone_name="prithvi_eo_v2_300",
            num_classes=2,
            lora_config=lora_cfg,
            pretrained=False,
        )
        params = model.count_parameters()
        assert params["trainable_pct"] < 10.0  # LoRA should be <10% trainable

    def test_forward_rgb(self):
        from models.prithvi.backbone import PrithviClassifier
        model = PrithviClassifier(
            backbone_name="prithvi_eo_v2_300",
            num_classes=2,
            freeze_backbone=True,
            pretrained=False,
        )
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 2)

    def test_convenience_builder(self):
        from models.prithvi.backbone import build_prithvi_lora
        model = build_prithvi_lora(num_classes=2, lora_r=4, lora_alpha=8)
        assert model is not None


class TestResNet50FPNFreezing:
    """Test layer freezing and parameter management."""

    def test_early_layers_frozen_by_default(self):
        from models.baseline.resnet50_fpn import ResNet50FPN
        model = ResNet50FPN(num_classes=2, pretrained=False)
        for param in model.layer0.parameters():
            assert not param.requires_grad
        for param in model.layer1.parameters():
            assert not param.requires_grad

    def test_later_layers_trainable_by_default(self):
        from models.baseline.resnet50_fpn import ResNet50FPN
        model = ResNet50FPN(num_classes=2, pretrained=False)
        for param in model.layer2.parameters():
            assert param.requires_grad
        for param in model.layer3.parameters():
            assert param.requires_grad

    def test_freeze_early_layers_false_all_trainable(self):
        from models.baseline.resnet50_fpn import ResNet50FPN
        model = ResNet50FPN(num_classes=2, pretrained=False, freeze_early_layers=False)
        params = model.count_parameters()
        assert params["trainable"] == params["total"]

    def test_frozen_reduces_trainable_count(self):
        from models.baseline.resnet50_fpn import ResNet50FPN
        frozen = ResNet50FPN(num_classes=2, pretrained=False, freeze_early_layers=True)
        unfrozen = ResNet50FPN(num_classes=2, pretrained=False, freeze_early_layers=False)
        assert frozen.count_parameters()["trainable"] < unfrozen.count_parameters()["trainable"]

    def test_different_num_classes(self):
        from models.baseline.resnet50_fpn import ResNet50FPN
        model = ResNet50FPN(num_classes=5, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 5)

    def test_output_is_finite(self):
        from models.baseline.resnet50_fpn import ResNet50FPN
        model = ResNet50FPN(num_classes=2, pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()

    def test_custom_fpn_channels(self):
        from models.baseline.resnet50_fpn import ResNet50FPN
        model = ResNet50FPN(num_classes=2, pretrained=False, fpn_out_channels=128)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)

    def test_trainable_pct_in_dict(self):
        from models.baseline.resnet50_fpn import ResNet50FPN
        model = ResNet50FPN(num_classes=2, pretrained=False)
        params = model.count_parameters()
        assert "trainable_pct" in params
        assert 0.0 < params["trainable_pct"] < 100.0


class TestLoRAConfigDefaults:
    """Test LoRA config dataclass fields beyond what TestPrithviLoRAConfig covers."""

    def test_dropout_default(self):
        from models.prithvi.backbone import LoRAConfig
        cfg = LoRAConfig()
        assert cfg.lora_dropout == 0.1

    def test_bias_default(self):
        from models.prithvi.backbone import LoRAConfig
        cfg = LoRAConfig()
        assert cfg.bias == "none"

    def test_v_linear_in_default_targets(self):
        from models.prithvi.backbone import LoRAConfig
        cfg = LoRAConfig()
        assert "qkv.v_linear" in cfg.target_modules

    def test_fc2_in_default_targets(self):
        from models.prithvi.backbone import LoRAConfig
        cfg = LoRAConfig()
        assert "mlp.fc2" in cfg.target_modules

    def test_custom_dropout(self):
        from models.prithvi.backbone import LoRAConfig
        cfg = LoRAConfig(lora_dropout=0.05)
        assert cfg.lora_dropout == 0.05

    def test_target_modules_preserved_after_post_init(self):
        from models.prithvi.backbone import LoRAConfig
        modules = ["qkv.q_linear"]
        cfg = LoRAConfig(target_modules=modules)
        assert cfg.target_modules is modules
