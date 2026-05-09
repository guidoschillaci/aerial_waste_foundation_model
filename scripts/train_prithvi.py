"""
Train Prithvi-EO-2.0 with LoRA on AerialWaste binary classification.

Usage:
    # LoRA fine-tuning (recommended)
    python scripts/train_prithvi.py --config configs/lora/prithvi_lora_classification.yaml

    # Full fine-tuning
    python scripts/train_prithvi.py --config configs/lora/prithvi_lora_classification.yaml \
        --no-lora --lr 1e-5

    # Frozen backbone (linear probe)
    python scripts/train_prithvi.py --config configs/lora/prithvi_lora_classification.yaml \
        --no-lora --freeze-backbone
"""

import argparse
import json
import time
from pathlib import Path

import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)
from tqdm import tqdm

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.dataset import AerialWasteDataset
from models.prithvi.backbone import LoRAConfig, PrithviClassifier


def _load_yaml_defaults(config_path: str) -> dict:
    """Read a YAML config and flatten it to argparse-compatible defaults."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    defaults = {}
    m = cfg.get("model", {}).get("model_args", {})
    if "backbone" in m:
        defaults["backbone"] = m["backbone"]
    lora_kw = m.get("peft_config", {}).get("peft_config_kwargs", {})
    if "r" in lora_kw:
        defaults["lora_r"] = lora_kw["r"]
    if "lora_alpha" in lora_kw:
        defaults["lora_alpha"] = lora_kw["lora_alpha"]
    if "peft_config" not in m:
        defaults["no_lora"] = True
    if m.get("freeze_backbone", False):
        defaults["freeze_backbone"] = True

    t = cfg.get("task", {})
    if "lr" in t:
        defaults["lr"] = t["lr"]
    if "weight_decay" in t.get("optimizer_hparams", {}):
        defaults["weight_decay"] = t["optimizer_hparams"]["weight_decay"]

    d = cfg.get("data", {})
    if "batch_size" in d:
        defaults["batch_size"] = d["batch_size"]
    if "num_workers" in d:
        defaults["num_workers"] = d["num_workers"]
    if "image_size" in d:
        defaults["image_size"] = d["image_size"]

    tr = cfg.get("trainer", {})
    if "max_epochs" in tr:
        defaults["epochs"] = tr["max_epochs"]

    return defaults


def parse_args():
    p = argparse.ArgumentParser(description="Train Prithvi-EO-2.0 on AerialWaste")
    p.add_argument("--config", default=None, type=str,
                   help="YAML config file (CLI flags override config values)")
    p.add_argument("--data-dir",        default="data/processed", type=str)
    p.add_argument("--splits-dir",      default="data/splits",    type=str)
    p.add_argument("--output-dir",      default="results/metrics", type=str)
    p.add_argument("--checkpoint-dir",  default="checkpoints",    type=str)

    # Model
    p.add_argument("--backbone",      default="prithvi_eo_v2_300", type=str)
    p.add_argument("--no-lora",       action="store_true", help="Disable LoRA")
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--lora-r",        default=16,   type=int)
    p.add_argument("--lora-alpha",    default=16,   type=int)
    p.add_argument("--dropout",       default=0.1,  type=float)
    p.add_argument("--image-size",    default=224,  type=int)

    # Training
    p.add_argument("--epochs",        default=30,   type=int)
    p.add_argument("--batch-size",    default=16,   type=int)
    p.add_argument("--lr",            default=6e-5, type=float)
    p.add_argument("--weight-decay",  default=0.01, type=float)
    p.add_argument("--warmup-epochs", default=3,    type=int)
    p.add_argument("--focal-loss",    action="store_true", default=True)
    p.add_argument("--num-workers",   default=4,    type=int)

    # Source filtering
    p.add_argument("--source", default=None, choices=["agea", "wv3", "ge"],
                   help="Train/eval on a single source only")

    p.add_argument("--seed", default=42, type=int)

    # Two-pass: first grab --config, then re-parse with YAML defaults applied
    known, _ = p.parse_known_args()
    if known.config:
        p.set_defaults(**_load_yaml_defaults(known.config))

    return p.parse_args()


class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced binary classification."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def get_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for batch in tqdm(loader, desc="  train", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu()
        all_probs.append(probs)
        all_labels.append(labels.cpu())

    all_probs  = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    return total_loss / len(loader.dataset), all_probs, all_labels


@torch.no_grad()
def evaluate(model, loader, criterion, device, prefix="val"):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels, all_sources, all_files = [], [], [], []

    for batch in tqdm(loader, desc=f"  {prefix}", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu()
        all_probs.append(probs)
        all_labels.append(labels.cpu())
        all_sources.extend(batch["source"])
        all_files.extend(batch["file"])

    all_probs  = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    ap_metric = BinaryAveragePrecision()
    f1_metric = BinaryF1Score()
    prec_metric = BinaryPrecision()
    rec_metric  = BinaryRecall()

    metrics = {
        "loss":      total_loss / len(loader.dataset),
        "AP":        ap_metric(all_probs, all_labels).item(),
        "F1":        f1_metric(all_probs, all_labels).item(),
        "precision": prec_metric(all_probs, all_labels).item(),
        "recall":    rec_metric(all_probs, all_labels).item(),
    }

    # Per-source AP
    for source in ["agea", "wv3", "ge"]:
        mask = torch.tensor([s == source for s in all_sources])
        if mask.sum() > 0:
            metrics[f"AP_{source}"] = ap_metric(
                all_probs[mask], all_labels[mask]
            ).item()

    return metrics, all_probs, all_labels, all_sources, all_files


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nDevice: {device}")

    # ── Load normalization stats ──────────────────────────────────────────────
    stats_path = Path(args.splits_dir) / "normalization_stats.json"
    norm_stats = None
    if stats_path.exists():
        with open(stats_path) as f:
            norm_stats = json.load(f)["norm_stats"]
        print(f"Normalization stats loaded from {stats_path}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    image_size = (args.image_size, args.image_size)
    ds_kwargs = dict(
        images_dir=Path(args.data_dir),  # will be overridden per split below
        task="classification",
        source_filter=args.source,
        norm_stats=norm_stats,
        image_size=image_size,
    )

    print("\nLoading datasets ...")
    train_ds = AerialWasteDataset(
        Path(args.splits_dir) / "train.json",
        images_dir=Path(args.data_dir) / "train",
        **{k: v for k, v in ds_kwargs.items() if k != "images_dir"},
    )
    val_ds = AerialWasteDataset(
        Path(args.splits_dir) / "val.json",
        images_dir=Path(args.data_dir) / "val",
        **{k: v for k, v in ds_kwargs.items() if k != "images_dir"},
    )
    test_ds = AerialWasteDataset(
        Path(args.splits_dir) / "test.json",
        images_dir=Path(args.data_dir) / "test",
        **{k: v for k, v in ds_kwargs.items() if k != "images_dir"},
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=AerialWasteDataset.collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=AerialWasteDataset.collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=AerialWasteDataset.collate_fn,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    lora_cfg = None if args.no_lora else LoRAConfig(r=args.lora_r, lora_alpha=args.lora_alpha)

    print("\nBuilding model ...")
    model = PrithviClassifier(
        backbone_name=args.backbone,
        num_classes=2,
        task="classification",
        freeze_backbone=args.freeze_backbone,
        lora_config=lora_cfg,
        dropout=args.dropout,
    ).to(device)

    param_info = model.count_parameters()
    print(f"Parameters: {param_info['total']:,} total | "
          f"{param_info['trainable']:,} trainable ({param_info['trainable_pct']:.2f}%)")

    # ── Loss ──────────────────────────────────────────────────────────────────
    class_weights = train_ds.class_weights.to(device)
    if args.focal_loss:
        criterion = FocalLoss(weight=class_weights)
        print("Loss: Focal loss (handles class imbalance)")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_tag = "lora" if not args.no_lora else ("frozen" if args.freeze_backbone else "full_ft")
    source_tag = args.source or "all"
    backbone_tag = args.backbone.split("_")[-1] + "m"  # "300m" or "600m"
    run_name = f"prithvi_{backbone_tag}_{mode_tag}_{source_tag}"

    best_ckpt   = ckpt_dir / f"{run_name}_best.pt"
    latest_ckpt = ckpt_dir / f"{run_name}_latest.pt"

    history  = []
    best_ap  = 0.0
    start_epoch = 1

    if latest_ckpt.exists():
        print(f"\nResuming from {latest_ckpt} ...")
        resume = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        history     = resume["history"]
        best_ap     = resume["best_ap"]
        start_epoch = resume["epoch"] + 1
        print(f"  Resumed at epoch {start_epoch} (best val AP so far: {best_ap:.4f})")

    print(f"\nTraining: {run_name}")
    print(f"Epochs: {start_epoch}–{args.epochs} | Batch: {args.batch_size} | LR: {args.lr}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, _, _ = train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        val_metrics, _, _, _, _ = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        ap = val_metrics["AP"]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_AP={ap:.4f} | "
            f"val_F1={val_metrics['F1']:.4f} | "
            f"time={elapsed:.1f}s"
        )

        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)

        if ap > best_ap:
            best_ap = ap
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "args": vars(args),
            }, best_ckpt)
            print(f"  ✓ Best model saved (AP={best_ap:.4f})")

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "history": history,
            "best_ap": best_ap,
            "args": vars(args),
        }, latest_ckpt)

    # ── Test evaluation ───────────────────────────────────────────────────────
    print(f"\nLoading best checkpoint ({best_ckpt}) for test evaluation ...")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_metrics, test_probs, test_labels, test_sources, test_files = evaluate(
        model, test_loader, criterion, device, prefix="test"
    )

    print("\n── Test Results ──────────────────────────────────────────────")
    for k, v in test_metrics.items():
        print(f"  {k:20s}: {v:.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "run_name":    run_name,
        "args":        vars(args),
        "param_info":  param_info,
        "best_val_AP": best_ap,
        "test_metrics": test_metrics,
        "history":     history,
    }

    # Save per-image predictions for downstream comparison
    preds_path = output_dir / f"{run_name}_test_preds.json"
    preds_data = {
        "model": run_name,
        "predictions": [
            {
                "file": f,
                "source": src,
                "prob_positive": float(prob),
                "label": int(lbl),
            }
            for f, src, prob, lbl in zip(test_files, test_sources, test_probs.tolist(), test_labels.tolist())
        ]
    }
    with open(preds_path, "w") as f:
        json.dump(preds_data, f, indent=2)

    results_path = output_dir / f"{run_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(f"Predictions saved to {preds_path}")
    print(f"\nDone. Best val AP: {best_ap:.4f}\n")


if __name__ == "__main__":
    main()
