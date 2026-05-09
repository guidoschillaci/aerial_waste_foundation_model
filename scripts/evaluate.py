"""
Evaluate all models and generate comparison report vs. AerialWaste paper baseline.

Loads result JSON files from results/metrics/, computes all metrics,
and generates publication-quality comparison plots.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --results-dir results/metrics --output results/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
)

# ── Paper-reported reference values (Torres & Fraternali, 2023) ──────────────
# AP AGEA (0.945) is paper-reported but not reproducible here — AerialWaste
# filenames are plain integers, so per-source grouping is unavailable.
PAPER_RESULTS = {
    "ResNet50+FPN (paper)": {
        "AP_all":    0.8799,
        "F1":        0.8070,
        "precision": 0.8189,
        "recall":    0.7954,
        "note":      "reported on 75/25 split (Torres & Fraternali, 2023); not directly comparable — different split from ours",
    }
}

MODEL_DISPLAY_NAMES = {
    "resnet50_fpn_paper_all":  "ResNet50+FPN (paper ckpt)",
    "resnet50_fpn_all":        "ResNet50+FPN (reproduced)",
    "prithvi_300m_frozen_all": "Prithvi-300M linear probe",
    "prithvi_600m_frozen_all": "Prithvi-600M linear probe",
    "prithvi_lora_all":        "Prithvi-300M + LoRA",
    "prithvi_300m_lora_all":   "Prithvi-300M + LoRA",
    "prithvi_600m_lora_all":   "Prithvi-600M + LoRA",
}

# Canonical display order: ResNets first, then Prithvi by size/mode
DISPLAY_ORDER = [
    "ResNet50+FPN (paper)",
    "ResNet50+FPN (paper ckpt)",
    "ResNet50+FPN (reproduced)",
    "Prithvi-300M linear probe",
    "Prithvi-600M linear probe",
    "Prithvi-300M + LoRA",
    "Prithvi-600M + LoRA",
]

COLORS = {
    "ResNet50+FPN (paper)":      "#922b21",   # deep red  — paper reference
    "ResNet50+FPN (paper ckpt)": "#e74c3c",   # red       — paper checkpoint
    "ResNet50+FPN (reproduced)": "#e59866",   # orange    — our reproduced baseline
    "Prithvi-300M linear probe": "#7fb3d3",   # light blue — frozen small
    "Prithvi-600M linear probe": "#2471a3",   # blue       — frozen large
    "Prithvi-300M + LoRA":       "#82e0aa",   # light green — LoRA small
    "Prithvi-600M + LoRA":       "#1e8449",   # dark green  — LoRA large
}


def _sort_by_display_order(d: dict) -> dict:
    order_map = {name: i for i, name in enumerate(DISPLAY_ORDER)}
    return dict(sorted(d.items(), key=lambda kv: order_map.get(kv[0], 999)))


def load_all_results(results_dir: Path) -> dict:
    """Load all *_results.json files from results directory."""
    results = {}
    for path in sorted(results_dir.glob("*_results.json")):
        with open(path) as f:
            data = json.load(f)
        run_name = data.get("run_name", path.stem.replace("_results", ""))
        display = MODEL_DISPLAY_NAMES.get(run_name, run_name)
        results[display] = data
    return results


def load_all_predictions(results_dir: Path) -> dict:
    """Load per-image predictions for PR curve plotting."""
    preds = {}
    for path in sorted(results_dir.glob("*_test_preds.json")):
        with open(path) as f:
            data = json.load(f)
        run_name = data.get("model", path.stem.replace("_test_preds", ""))
        display = MODEL_DISPLAY_NAMES.get(run_name, run_name)
        preds[display] = data.get("predictions", [])
    return preds


def build_comparison_table(all_results: dict) -> pd.DataFrame:
    """Build a unified comparison DataFrame."""
    rows = []

    # Paper reference
    for name, ref in PAPER_RESULTS.items():
        rows.append({
            "Model":       name,
            "AP (all)":    ref["AP_all"],
            "F1":          ref["F1"],
            "Precision":   ref.get("precision"),
            "Recall":      ref.get("recall"),
            "Trainable %": "100%",
            "Note":        ref.get("note", ""),
        })

    # Our results
    for display_name, data in all_results.items():
        metrics = data.get("test_metrics", {})
        params  = data.get("param_info", {})
        trainable_pct = params.get("trainable_pct", 100.0)

        rows.append({
            "Model":       display_name,
            "AP (all)":    metrics.get("AP"),
            "F1":          metrics.get("F1"),
            "Precision":   metrics.get("precision"),
            "Recall":      metrics.get("recall"),
            "Trainable %": f"{trainable_pct:.1f}%",
            "Note":        "this work",
        })

    df = pd.DataFrame(rows)
    # Sort rows by canonical display order
    order_map = {name: i for i, name in enumerate(DISPLAY_ORDER)}
    df["_order"] = df["Model"].map(lambda x: order_map.get(x, 999))
    df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return df


def plot_ap_comparison(df: pd.DataFrame, output_dir: Path):
    """Bar chart comparing Average Precision across models."""
    fig, ax = plt.subplots(figsize=(9, 5))

    valid = df[df["AP (all)"].notna()].copy()
    colors = [COLORS.get(name, "#7f8c8d") for name in valid["Model"]]
    bars = ax.barh(valid["Model"], valid["AP (all)"], color=colors, edgecolor="white", height=0.6)

    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("Average Precision", fontsize=11)
    ax.set_title("AerialWaste Detection — Average Precision", fontsize=13, fontweight="bold")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)

    for bar, val in zip(bars, valid["AP (all)"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    paper_patch = mpatches.Patch(color=COLORS["ResNet50+FPN (paper)"],
                                  label="Torres & Fraternali (2023) reference")
    lora_patch  = mpatches.Patch(color=COLORS.get("Prithvi-600M + LoRA", "#27ae60"),
                                  label="Ours (Prithvi + LoRA)")
    ax.legend(handles=[paper_patch, lora_patch], fontsize=9, loc="lower right")

    plt.tight_layout()
    out = output_dir / "ap_comparison.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_pr_curves(all_preds: dict, output_dir: Path):
    """Precision-Recall curves — excludes paper checkpoint (different split / sigmoid calibration)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, preds in all_preds.items():
        if model_name == "ResNet50+FPN (paper ckpt)":
            continue
        if len(preds) < 10:
            continue
        y_true = [p["label"]        for p in preds]
        y_prob = [p["prob_positive"] for p in preds]
        try:
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            color = COLORS.get(model_name, None)
            ax.plot(rec, prec, label=f"{model_name} (AP={ap:.3f})", color=color, linewidth=2)
        except Exception:
            continue

    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_title("Precision-Recall Curves", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = output_dir / "pr_curves.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_training_curves(all_results: dict, output_dir: Path, max_epochs: int = 20):
    """Training loss and val AP curves per model, capped at max_epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model_name, data in all_results.items():
        history = data.get("history", [])
        if not history:
            continue
        history = [h for h in history if h["epoch"] <= max_epochs]
        if not history:
            continue
        epochs     = [h["epoch"]     for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_ap     = [h.get("val_AP", h.get("val_ap")) for h in history]

        color = COLORS.get(model_name, None)
        axes[0].plot(epochs, train_loss, label=model_name, color=color, linewidth=2)
        if any(v is not None for v in val_ap):
            axes[1].plot(epochs, val_ap, label=model_name, color=color, linewidth=2)

    for ax, ylabel, title in zip(axes,
        ["Training Loss", "Validation AP"],
        ["Training Loss", "Validation Average Precision"]):
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlim(1, max_epochs)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = output_dir / "training_curves.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_roc_curves(all_preds: dict, output_dir: Path):
    """ROC curves with AUC for all models."""
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    for model_name, preds in all_preds.items():
        if model_name == "ResNet50+FPN (paper ckpt)":
            continue
        if len(preds) < 10:
            continue
        y_true = [p["label"]        for p in preds]
        y_prob = [p["prob_positive"] for p in preds]
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            color = COLORS.get(model_name, None)
            ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})", color=color, linewidth=2)
        except Exception:
            continue

    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("ROC Curves", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = output_dir / "roc_curves.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_param_efficiency(all_results: dict, output_dir: Path):
    """Scatter: AP vs trainable parameter count — shows LoRA efficiency."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Paper reference point
    ax.scatter([100.0], [87.9], color=COLORS["ResNet50+FPN (paper)"],
               s=160, zorder=5, marker="*")
    ax.annotate("ResNet50+FPN\n(paper)", (100.0, 87.9),
                textcoords="offset points", xytext=(6, -14), fontsize=8)

    for display_name, data in all_results.items():
        metrics = data.get("test_metrics", {})
        params  = data.get("param_info", {})
        ap  = metrics.get("AP")
        pct = params.get("trainable_pct")
        if ap is None or pct is None:
            continue
        color = COLORS.get(display_name, "#7f8c8d")
        ax.scatter([pct], [ap * 100], color=color, s=120, zorder=5)
        ax.annotate(display_name, (pct, ap * 100),
                    textcoords="offset points", xytext=(6, 4), fontsize=7.5)

    ax.set_xlabel("Trainable Parameters (%)", fontsize=10)
    ax.set_ylabel("Test AP (%)", fontsize=10)
    ax.set_title("Parameter Efficiency", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = output_dir / "param_efficiency.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def _optimal_threshold(y_true, y_prob) -> float:
    """Threshold that maximises F1 on the given predictions."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / np.where((prec + rec) == 0, 1, prec + rec)
    idx = np.argmax(f1[:-1])   # last entry has no threshold
    return float(thresholds[idx])


def plot_confusion_matrices(all_preds: dict, output_dir: Path):
    """Two rows of confusion matrices: fixed 0.5 threshold and optimal F1 threshold.

    Excludes the paper checkpoint: its training data likely overlaps our test split
    (different 75/25 vs our 70/15/15 split), making threshold-based metrics unreliable
    for direct comparison.
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Exclude paper checkpoint — train/test overlap makes threshold metrics unfair
    preds_filtered = {k: v for k, v in all_preds.items()
                      if k != "ResNet50+FPN (paper ckpt)"}

    n = len(preds_filtered)
    if n == 0:
        return

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (model_name, preds) in enumerate(preds_filtered.items()):
        y_true = [p["label"]        for p in preds]
        y_prob = [p["prob_positive"] for p in preds]

        for row, (thresh, label) in enumerate([
            (0.5,                                   "threshold = 0.5"),
            (_optimal_threshold(y_true, y_prob),    "optimal threshold"),
        ]):
            y_pred = [int(p >= thresh) for p in y_prob]
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(cm, display_labels=["Clean", "Waste"])
            disp.plot(ax=axes[row, col], colorbar=False, cmap="Blues")
            title = model_name if row == 0 else f"(τ={thresh:.2f})"
            axes[row, col].set_title(title, fontsize=8, fontweight="bold" if row == 0 else "normal")

    axes[0, 0].set_ylabel("Fixed threshold = 0.5", fontsize=9)
    axes[1, 0].set_ylabel("Optimal F1 threshold", fontsize=9)

    plt.suptitle("Confusion Matrices", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = output_dir / "confusion_matrices.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare all models")
    parser.add_argument("--results-dir", default="results/metrics", type=str)
    parser.add_argument("--output",      default="results/figures",  type=str)
    parser.add_argument("--no-plots",    action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nAerialWaste Foundation Model Evaluation")
    print("=" * 55)

    # ── Load results ──────────────────────────────────────────────────────────
    all_results = _sort_by_display_order(load_all_results(results_dir))
    all_preds   = _sort_by_display_order(load_all_predictions(results_dir))

    if not all_results:
        print(f"\n[warn] No *_results.json files found in {results_dir}")
        print("Run train_baseline.py and train_prithvi.py first.")
        return

    print(f"\nLoaded results for {len(all_results)} model(s):")
    for name in all_results:
        print(f"  - {name}")

    # ── Comparison table ──────────────────────────────────────────────────────
    df = build_comparison_table(all_results)

    print("\n── Results Comparison Table ─────────────────────────────────")
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    print(df.to_string(index=False))

    # Save table
    csv_path = output_dir / "comparison_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nTable saved to {csv_path}")

    # Save markdown table
    md_path = output_dir / "comparison_table.md"
    with open(md_path, "w") as f:
        f.write("# AerialWaste Detection Benchmark Results\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n*Reference: Torres & Fraternali, Scientific Data 2023*\n")
    print(f"Markdown table saved to {md_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        print("\nGenerating plots ...")
        plt.rcParams.update({
            "font.family":  "sans-serif",
            "font.size":    10,
            "figure.dpi":   150,
        })

        plot_ap_comparison(df, output_dir)

        if all_preds:
            plot_pr_curves(all_preds, output_dir)
            plot_roc_curves(all_preds, output_dir)
            plot_confusion_matrices(all_preds, output_dir)

        if any(data.get("history") for data in all_results.values()):
            plot_training_curves(all_results, output_dir)

        plot_param_efficiency(all_results, output_dir)

    print(f"\nAll outputs saved to {output_dir}/\n")


if __name__ == "__main__":
    main()
