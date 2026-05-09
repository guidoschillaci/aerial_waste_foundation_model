"""
Evaluate the original AerialWaste paper checkpoint (Torres & Fraternali, 2023).

Loads checkpoint.pth from the authors' Google Drive, reconstructs their exact
ResNet50+FPN architecture, and runs evaluation on our test split.

Usage:
    python scripts/eval_paper_baseline.py
    python scripts/eval_paper_baseline.py \
        --checkpoint checkpoints/aerialwaste_paper/aerialwaste-model/checkpoint.pth
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    BinaryAveragePrecision, BinaryF1Score, BinaryPrecision, BinaryRecall,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.dataset import AerialWasteDataset


class PaperResNet50FPN(nn.Module):
    """
    Exact reconstruction of the Torres & Fraternali (2023) model.

    FPN with dense top-down concatenation (not addition):
      - p4 = cat(upsample(p5), lat(c4))         → 512 ch → smooth1 → 256
      - p3 = cat(upsample(p4_raw), lat(c3))      → 768 ch → smooth2 → 256
      - p2 = cat(upsample(p3_raw), lat(c2))      → 1024 ch → smooth3 → 256
    Each level is GAP-pooled, fed through shared fc, then aggregated by classifier.
    Output is a single binary logit (BCE loss in training).
    """

    def __init__(self):
        super().__init__()
        resnet = tv_models.resnet50(weights=None)

        self.stage0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.stage1 = nn.Sequential(resnet.layer1)
        self.stage2 = nn.Sequential(resnet.layer2)
        self.stage3 = nn.Sequential(resnet.layer3)
        self.stage4 = nn.Sequential(resnet.layer4)

        self.toplayer  = nn.Conv2d(2048, 256, kernel_size=1)
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1)

        self.smooth1 = nn.Conv2d( 512, 256, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d( 768, 256, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(1024, 256, kernel_size=3, padding=1)

        self.fc         = nn.Linear(256, 1)
        self.classifier = nn.Linear(4, 1)

        # ModuleLists that hold the same references (match checkpoint key names)
        self.backbone    = nn.ModuleList([self.stage0, self.stage1, self.stage2,
                                          self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.toplayer, self.latlayer1, self.latlayer2,
                                          self.latlayer3, self.smooth1, self.smooth2,
                                          self.smooth3, self.fc, self.classifier])

    @staticmethod
    def _upsample_cat(x, y):
        _, _, H, W = y.size()
        return torch.cat([F.interpolate(x, size=(H, W), mode="nearest"), y], dim=1)

    def forward(self, x):
        c1 = self.stage0(x)
        c2 = self.stage1(c1)
        c3 = self.stage2(c2)   # detached during training; irrelevant at eval
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)

        p5      = self.toplayer(c5)
        p4_raw  = self._upsample_cat(p5, self.latlayer1(c4))
        p3_raw  = self._upsample_cat(p4_raw, self.latlayer2(c3))
        p2_raw  = self._upsample_cat(p3_raw, self.latlayer3(c2))

        p4 = self.smooth1(p4_raw)
        p3 = self.smooth2(p3_raw)
        p2 = self.smooth3(p2_raw)

        def gap(t):
            return t.mean(dim=[2, 3])

        out5 = F.relu(self.fc(gap(p5)))
        out4 = F.relu(self.fc(gap(p4)))
        out3 = F.relu(self.fc(gap(p3)))
        out2 = F.relu(self.fc(gap(p2)))

        out = torch.cat([out5, out4, out3, out2], dim=1)  # [B, 4]
        return self.classifier(out)                        # [B, 1]


def load_paper_model(ckpt_path: str, device: torch.device) -> PaperResNet50FPN:
    model = PaperResNet50FPN()
    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] missing keys: {missing[:5]}")
    if unexpected:
        print(f"  [warn] unexpected keys: {unexpected[:5]}")
    return model.to(device).eval()


@torch.no_grad()
def evaluate(model, loader, device):
    all_probs, all_labels, all_sources = [], [], []

    for batch in tqdm(loader, desc="  eval", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"]
        logits = model(images).squeeze(1).cpu()   # [B]
        probs  = torch.sigmoid(logits)
        all_probs.append(probs)
        all_labels.append(labels)
        all_sources.extend(batch["source"])

    all_probs  = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    ap_m   = BinaryAveragePrecision()
    f1_m   = BinaryF1Score()
    prec_m = BinaryPrecision()
    rec_m  = BinaryRecall()

    return {
        "AP":        ap_m(all_probs, all_labels).item(),
        "F1":        f1_m(all_probs, all_labels).item(),
        "precision": prec_m(all_probs, all_labels).item(),
        "recall":    rec_m(all_probs, all_labels).item(),
    }, all_probs, all_labels, all_sources


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="checkpoints/aerialwaste_paper/aerialwaste-model/checkpoint.pth")
    p.add_argument("--splits-dir",  default="data/splits")
    p.add_argument("--data-dir",    default="data/processed")
    p.add_argument("--output-dir",  default="results/metrics")
    p.add_argument("--image-size",  default=800, type=int)
    p.add_argument("--batch-size",  default=4, type=int)
    p.add_argument("--num-workers", default=4, type=int)
    args = p.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nDevice: {device}")

    # Paper model was trained from ImageNet-pretrained ResNet50 — use ImageNet stats,
    # not our AerialWaste-computed stats which differ significantly.
    norm_stats = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

    image_size = (args.image_size, args.image_size)
    test_ds = AerialWasteDataset(
        Path(args.splits_dir) / "test.json",
        images_dir=Path(args.data_dir) / "test",
        norm_stats=norm_stats,
        image_size=image_size,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=AerialWasteDataset.collate_fn,
        pin_memory=device.type == "cuda",
    )

    print(f"\nLoading paper checkpoint from {args.checkpoint} ...")
    model = load_paper_model(args.checkpoint, device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")

    print("\nEvaluating on test set ...")
    metrics, probs, labels, sources = evaluate(model, test_loader, device)

    print("\n── Test Results (paper model) ────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")

    run_name = "resnet50_fpn_paper_all"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preds_path = output_dir / f"{run_name}_test_preds.json"
    with open(preds_path, "w") as f:
        json.dump({
            "model": run_name,
            "predictions": [
                {"source": s, "prob_positive": float(p), "label": int(l)}
                for s, p, l in zip(sources, probs.tolist(), labels.tolist())
            ]
        }, f, indent=2)

    results_path = output_dir / f"{run_name}_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "run_name":     run_name,
            "test_metrics": metrics,
            "param_info":   {"total": total, "trainable": total, "trainable_pct": 100.0},
            "note":         "Original checkpoint from Torres & Fraternali (2023), evaluated on our test split",
        }, f, indent=2)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
