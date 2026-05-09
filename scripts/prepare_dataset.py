"""
Prepare AerialWaste dataset for training.

Steps:
  1. Parse COCO-format JSON annotations (training.json, testing.json)
  2. Copy/symlink images into processed/ with consistent naming
  3. Generate per-image binary label files and multi-label vectors
  4. Compute per-source normalization statistics (mean/std)
  5. Create train/val split from training.json (80/20 stratified by source)
  6. Write final split manifests to data/splits/

Usage:
    python scripts/prepare_dataset.py --input data/raw --output data/processed
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# AerialWaste source identifiers (appear in filenames or metadata)
SOURCES = ["agea", "wv3", "ge"]

# AerialWaste waste type categories (15 classes from aerialwaste.org)
WASTE_TYPES = [
    "rubble_excavated_earth",
    "bulky_items",
    "firewood",
    "scrap",
    "plastic",
    "vehicles",
    "tires",
    "domestic_appliances",
    "paper",
    "sludge_zootechnical_manure",
    "stone_marble_waste",
    "asphalt_milling",
    "corrugated_sheets_asbestos",
    "glass",
    "foundry_waste",
]

# Storage modes (7 classes)
# Note: These may not be mutually exclusive with waste types, so we will treat them as separate multi-label vectors.
STORAGE_MODES = [
    "heaps_not_delimited",
    "container",
    "big_bags",
    "pallets",
    "delimited_heaps",
    "cisterns",
    "drums_bins",
]


def load_coco_json(json_path: Path) -> tuple[list, list, list]:
    """Load COCO-format AerialWaste JSON. Returns (images, annotations, categories)."""
    with open(json_path) as f:
        data = json.load(f)
    return data["images"], data.get("annotations", []), data.get("categories", [])


def infer_source(image_info: dict) -> str:
    """Infer image source from filename or metadata."""
    fname = image_info.get("file_name", "").lower()
    if "agea" in fname or "orthophoto" in fname:
        return "agea"
    elif "wv3" in fname or "worldview" in fname:
        return "wv3"
    elif "ge" in fname or "google" in fname:
        return "ge"
    # Fall back to heuristic on image size (AGEA tends to be larger)
    return "unknown"


def compute_normalization_stats(image_paths: list[Path], sample_size: int = 500) -> dict:
    """Compute per-channel mean and std for normalization."""
    sample = image_paths[:sample_size] if len(image_paths) > sample_size else image_paths
    means, stds = [], []

    for p in tqdm(sample, desc="Computing normalization stats", leave=False):
        try:
            img = np.array(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
            means.append(img.mean(axis=(0, 1)))
            stds.append(img.std(axis=(0, 1)))
        except Exception:
            continue

    if not means:
        print("  [warn] No images loaded — using ImageNet defaults")
        return {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

    return {
        "mean": np.mean(means, axis=0).tolist(),
        "std": np.mean(stds, axis=0).tolist(),
    }


def build_multilabel_vector(categories: list[str], annotation_cats: list[str]) -> list[int]:
    """Build a binary multi-label vector for waste types."""
    # Initialize vector with zeros
    vec = [0] * len(categories)
    for cat in annotation_cats:
        # Normalize category name for matching (lowercase, replace spaces/slashes)
        cat_normalized = cat.lower().replace(" ", "_").replace("/", "_")
        for i, wt in enumerate(categories):
            # Match if category name contains the waste type or vice versa (to handle minor variations)
            if wt in cat_normalized or cat_normalized in wt:
                # Set corresponding index to 1 if there's a match
                vec[i] = 1
    return vec


def prepare_split(
    images: list[dict],
    annotations: list[dict],
    images_dir: Path,
    output_dir: Path,
    split_name: str,
) -> list[dict]:
    """Process one split: copy images, generate labels, return manifest."""
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # Build annotation lookup by image_id
    ann_by_img = defaultdict(list)
    for ann in annotations:
        ann_by_img[ann["image_id"]].append(ann)

    manifest = []
    missing = 0

    for img_info in tqdm(images, desc=f"Preparing {split_name}"):
        img_id = img_info["id"]
        fname = img_info["file_name"]
        is_positive = img_info.get("is_candidate_location", 0)
        source = infer_source(img_info)

        # Find image file
        src_path = images_dir / fname
        if not src_path.exists():
            # Try common alternative locations
            for candidate in images_dir.rglob(Path(fname).name):
                src_path = candidate
                break
            else:
                missing += 1
                continue

        # Copy to processed dir with canonical name
        canonical_name = f"{img_id:06d}_{source}.jpg"
        dest_path = split_dir / canonical_name
        if not dest_path.exists():
            try:
                img = Image.open(src_path).convert("RGB")
                img.save(dest_path, quality=95)
            except Exception as e:
                print(f"  [warn] Could not process {fname}: {e}")
                missing += 1
                continue

        # Build multi-label vector for waste types
        img_anns = ann_by_img.get(img_id, [])
        waste_cats = []
        storage_cats = []
        for ann in img_anns:
            cats = ann.get("categories", [])
            waste_cats.extend([c for c in cats if c in WASTE_TYPES])
            storage_cats.extend([c for c in cats if c in STORAGE_MODES])

        waste_vector = build_multilabel_vector(WASTE_TYPES, waste_cats)
        storage_vector = build_multilabel_vector(STORAGE_MODES, storage_cats)

        # Segmentation mask path (if available)
        has_mask = any("segmentation" in ann and ann["segmentation"] for ann in img_anns)

        manifest.append({
            "id": img_id,
            "file": canonical_name,
            "source": source,
            "label": int(is_positive),
            "evidence": img_info.get("evidence", -1),
            "severity": img_info.get("severity", -1),
            "site_type": img_info.get("site_type", ""),
            "waste_types": waste_vector,
            "storage_modes": storage_vector,
            "has_mask": has_mask,
            "width": img_info.get("width", 0),
            "height": img_info.get("height", 0),
        })

    if missing > 0:
        print(f"  [warn] {missing} images not found (check {images_dir}/)")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Prepare AerialWaste dataset")
    parser.add_argument("--input", type=str, default="data/raw",
                        help="Raw dataset directory (contains training.json, testing.json, images/)")
    parser.add_argument("--output", type=str, default="data/processed",
                        help="Output directory for processed dataset")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Fraction of training set to use for validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)

    images_dir = input_dir
    print(f"\nAerialWaste Dataset Preparation")
    print(f"Input:  {input_dir.resolve()}")
    print(f"Output: {output_dir.resolve()}\n")

    # ── Load annotations ─────────────────────────────────────────────────────
    print("Loading annotations ...")
    train_json = input_dir / "training.json"
    test_json = input_dir / "testing.json"

    if not train_json.exists():
        raise FileNotFoundError(f"training.json not found in {input_dir}. "
                                "Run scripts/download_dataset.py first.")

    train_images, train_anns, categories = load_coco_json(train_json)
    test_images, test_anns, _ = load_coco_json(test_json) if test_json.exists() else ([], [], [])

    print(f"  Training images: {len(train_images)} | annotations: {len(train_anns)}")
    print(f"  Test images:     {len(test_images)} | annotations: {len(test_anns)}")

    # ── Train/val split (stratified by label) ────────────────────────────────
    labels = [img.get("is_candidate_location", 0) for img in train_images]
    train_imgs, val_imgs = train_test_split(
        train_images,
        test_size=args.val_ratio,
        stratify=labels,
        random_state=args.seed,
    )
    print(f"\nSplit: {len(train_imgs)} train | {len(val_imgs)} val | {len(test_images)} test")

    # ── Get corresponding annotations for val split ───────────────────────────
    val_ids = {img["id"] for img in val_imgs}
    train_ids = {img["id"] for img in train_imgs}
    val_anns = [a for a in train_anns if a["image_id"] in val_ids]
    sub_train_anns = [a for a in train_anns if a["image_id"] in train_ids]

    # ── Process each split ───────────────────────────────────────────────────
    train_manifest = prepare_split(train_imgs, sub_train_anns, images_dir, output_dir, "train")
    val_manifest = prepare_split(val_imgs, val_anns, images_dir, output_dir, "val")
    test_manifest = prepare_split(test_images, test_anns, images_dir, output_dir, "test")

    # ── Compute normalization stats (on training set) ────────────────────────
    print("\nComputing normalization statistics ...")
    train_img_paths = [output_dir / "train" / m["file"] for m in train_manifest
                       if (output_dir / "train" / m["file"]).exists()]
    norm_stats = compute_normalization_stats(train_img_paths)
    print(f"  Mean: {[f'{v:.4f}' for v in norm_stats['mean']]}")
    print(f"  Std:  {[f'{v:.4f}' for v in norm_stats['std']]}")

    # ── Save manifests and stats ─────────────────────────────────────────────
    for name, manifest in [("train", train_manifest), ("val", val_manifest), ("test", test_manifest)]:
        out = splits_dir / f"{name}.json"
        with open(out, "w") as f:
            json.dump(manifest, f, indent=2)
        positives = sum(m["label"] for m in manifest)
        print(f"  Saved {out}: {len(manifest)} items ({positives} positive, "
              f"{len(manifest)-positives} negative)")

    stats_path = splits_dir / "normalization_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "norm_stats": norm_stats,
            "waste_types": WASTE_TYPES,
            "storage_modes": STORAGE_MODES,
            "sources": SOURCES,
            "split_sizes": {
                "train": len(train_manifest),
                "val": len(val_manifest),
                "test": len(test_manifest),
            }
        }, f, indent=2)
    print(f"  Saved {stats_path}")

    print("\nPreparation complete.\n")


if __name__ == "__main__":
    main()
