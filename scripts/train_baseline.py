"""
Train the ResNet50+FPN baseline on AerialWaste.

Reproduces the original paper results as closely as possible.
Saves metrics in the same format as train_prithvi.py for direct comparison.

Usage:
    python scripts/train_baseline.py
    python scripts/train_baseline.py --source agea  # AGEA-only training
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchmetrics.classification import (
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.dataset import AerialWasteDataset
from models.baseline.resnet50_fpn import ResNet50FPN


class RandomRotation90(torch.nn.Module):
    """Randomly rotate PIL image by 0°, 90°, 180°, or 270°."""
    def forward(self, img):
        angle = torch.randint(0, 4, (1,)).item() * 90
        return T.functional.rotate(img, angle)


def parse_args():
    p = argparse.ArgumentParser(description="Train ResNet50+FPN baseline on AerialWaste")
    p.add_argument("--data-dir",       default="data/processed", type=str)
    p.add_argument("--splits-dir",     default="data/splits",    type=str)
    p.add_argument("--output-dir",     default="results/metrics", type=str)
    p.add_argument("--checkpoint-dir", default="checkpoints",    type=str)
    p.add_argument("--epochs",         default=20,   type=int)
    p.add_argument("--batch-size",     default=2,    type=int)
    p.add_argument("--grad-accum",     default=6,    type=int,
                   help="Gradient accumulation steps (effective batch = batch-size * grad-accum)")
    p.add_argument("--lr",             default=5e-3, type=float)
    p.add_argument("--momentum",       default=0.9,  type=float)
    p.add_argument("--weight-decay",   default=1e-4, type=float)
    p.add_argument("--image-size",     default=800,  type=int)
    p.add_argument("--early-stop-patience", default=10, type=int)
    p.add_argument("--num-workers",    default=4,    type=int)
    p.add_argument("--source", default=None, choices=["agea", "wv3", "ge"])
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, criterion, device, prefix="val"):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels, all_sources = [], [], []

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

    all_probs  = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    ap_metric   = BinaryAveragePrecision()
    f1_metric   = BinaryF1Score()
    prec_metric = BinaryPrecision()
    rec_metric  = BinaryRecall()

    metrics = {
        "loss":      total_loss / len(loader.dataset),
        "AP":        ap_metric(all_probs, all_labels).item(),
        "F1":        f1_metric(all_probs, all_labels).item(),
        "precision": prec_metric(all_probs, all_labels).item(),
        "recall":    rec_metric(all_probs, all_labels).item(),
    }
    for source in ["agea", "wv3", "ge"]:
        mask = torch.tensor([s == source for s in all_sources])
        if mask.sum() > 0:
            metrics[f"AP_{source}"] = ap_metric(all_probs[mask], all_labels[mask]).item()

    return metrics, all_probs, all_labels, all_sources


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

    stats_path = Path(args.splits_dir) / "normalization_stats.json"
    norm_stats = None
    if stats_path.exists():
        with open(stats_path) as f:
            norm_stats = json.load(f)["norm_stats"]

    image_size = (args.image_size, args.image_size)
    mean = norm_stats["mean"] if norm_stats else [0.485, 0.456, 0.406]
    std  = norm_stats["std"]  if norm_stats else [0.229, 0.224, 0.225]

    # Paper augmentation: flips + 90° rotations
    train_transform = T.Compose([
        T.Resize(image_size, antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        RandomRotation90(),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    print("\nLoading datasets ...")
    train_ds = AerialWasteDataset(
        Path(args.splits_dir) / "train.json",
        images_dir=Path(args.data_dir) / "train",
        source_filter=args.source,
        norm_stats=norm_stats,
        image_size=image_size,
        transform=train_transform,
    )
    val_ds = AerialWasteDataset(
        Path(args.splits_dir) / "val.json",
        images_dir=Path(args.data_dir) / "val",
        source_filter=args.source,
        norm_stats=norm_stats,
        image_size=image_size,
    )
    test_ds = AerialWasteDataset(
        Path(args.splits_dir) / "test.json",
        images_dir=Path(args.data_dir) / "test",
        source_filter=args.source,
        norm_stats=norm_stats,
        image_size=image_size,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=AerialWasteDataset.collate_fn,
                              pin_memory=device.type == "cuda")
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=AerialWasteDataset.collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=AerialWasteDataset.collate_fn)

    print("\nBuilding ResNet50+FPN model ...")
    model = ResNet50FPN(num_classes=2, pretrained=True).to(device)
    params = model.count_parameters()
    print(f"Parameters: {params['total']:,} total | {params['trainable']:,} trainable")

    class_weights = train_ds.class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir   = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_tag = args.source or "all"
    run_name   = f"resnet50_fpn_{source_tag}"
    best_ap        = 0.0
    best_ckpt      = ckpt_dir / f"{run_name}_best.pt"
    history        = []
    epochs_no_improve = 0

    effective_batch = args.batch_size * args.grad_accum
    print(f"\nTraining: {run_name} | {args.epochs} epochs | "
          f"batch={args.batch_size}×{args.grad_accum}={effective_batch} | "
          f"early stop patience={args.early_stop_patience}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc="  train", leave=False)):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            loss = criterion(model(images), labels) / args.grad_accum
            loss.backward()
            train_loss += loss.item() * args.grad_accum * images.size(0)

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()
        train_loss /= len(train_loader.dataset)
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_AP={val_metrics['AP']:.4f} | "
            f"val_F1={val_metrics['F1']:.4f} | "
            f"time={time.time()-t0:.1f}s"
        )
        history.append({"epoch": epoch, "train_loss": train_loss,
                         **{f"val_{k}": v for k, v in val_metrics.items()}})

        if val_metrics["AP"] > best_ap + 5e-4:
            best_ap = val_metrics["AP"]
            epochs_no_improve = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                         "val_metrics": val_metrics, "args": vars(args)}, best_ckpt)
            print(f"  ✓ Best model saved (AP={best_ap:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.early_stop_patience} epochs)")
                break

    print(f"\nLoading best checkpoint for test evaluation ...")
    model.load_state_dict(torch.load(best_ckpt, map_location=device)["model_state"])
    test_metrics, test_probs, test_labels, test_sources = evaluate(
        model, test_loader, criterion, device, prefix="test"
    )

    print("\n── Test Results ──────────────────────────────────────────────")
    for k, v in test_metrics.items():
        print(f"  {k:20s}: {v:.4f}")

    results = {
        "run_name":     run_name,
        "args":         vars(args),
        "param_info":   params,
        "best_val_AP":  best_ap,
        "test_metrics": test_metrics,
        "history":      history,
        # Paper-reported reference values for comparison
        "paper_reference": {
            "AP_all_sources": 0.879,
            "AP_agea":        0.945,
        }
    }

    preds_path = output_dir / f"{run_name}_test_preds.json"
    with open(preds_path, "w") as f:
        json.dump({
            "model": run_name,
            "predictions": [
                {"source": s, "prob_positive": float(p), "label": int(l)}
                for s, p, l in zip(test_sources, test_probs.tolist(), test_labels.tolist())
            ]
        }, f, indent=2)

    results_path = output_dir / f"{run_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults → {results_path}\nPredictions → {preds_path}\n")


if __name__ == "__main__":
    main()
