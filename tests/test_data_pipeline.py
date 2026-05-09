"""Tests for the AerialWaste data pipeline."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.dataset import AerialWasteDataset


@pytest.fixture
def tmp_dataset(tmp_path):
    """Create a minimal fake AerialWaste dataset for testing."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    manifest = []
    for i in range(20):
        img = Image.fromarray(
            (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        )
        fname = f"{i:06d}_{'agea' if i < 7 else 'wv3' if i < 14 else 'ge'}.jpg"
        img.save(images_dir / fname)

        manifest.append({
            "id": i,
            "file": fname,
            "source": "agea" if i < 7 else ("wv3" if i < 14 else "ge"),
            "label": 1 if i % 3 == 0 else 0,
            "evidence": 2,
            "severity": 1,
            "site_type": "production_area",
            "waste_types": [0] * 15,
            "storage_modes": [0] * 7,
            "has_mask": False,
        })

    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    return manifest_path, images_dir, manifest


def test_dataset_loads(tmp_dataset):
    manifest_path, images_dir, manifest = tmp_dataset
    ds = AerialWasteDataset(manifest_path, images_dir, task="classification")
    assert len(ds) == len(manifest)


def test_dataset_item_structure(tmp_dataset):
    manifest_path, images_dir, _ = tmp_dataset
    ds = AerialWasteDataset(manifest_path, images_dir)
    item = ds[0]
    assert "image" in item
    assert "label" in item
    assert "source" in item
    assert "file" in item
    assert isinstance(item["image"], torch.Tensor)
    assert item["image"].shape == (3, 224, 224)


def test_dataset_label_types(tmp_dataset):
    manifest_path, images_dir, _ = tmp_dataset
    ds = AerialWasteDataset(manifest_path, images_dir, task="classification")
    item = ds[0]
    assert item["label"].dtype == torch.long
    assert item["label"].item() in (0, 1)


def test_dataset_multilabel(tmp_dataset):
    manifest_path, images_dir, _ = tmp_dataset
    ds = AerialWasteDataset(manifest_path, images_dir, task="multilabel")
    item = ds[0]
    assert item["label"].shape == (15,)
    assert item["label"].dtype == torch.float32


def test_dataset_source_filter(tmp_dataset):
    manifest_path, images_dir, _ = tmp_dataset
    ds_agea = AerialWasteDataset(manifest_path, images_dir, source_filter="agea")
    assert len(ds_agea) == 7
    for item in ds_agea:
        assert item["source"] == "agea"


def test_dataset_class_weights(tmp_dataset):
    manifest_path, images_dir, _ = tmp_dataset
    ds = AerialWasteDataset(manifest_path, images_dir)
    weights = ds.class_weights
    assert weights.shape == (2,)
    assert (weights > 0).all()
    # Positive class should have higher weight (minority)
    assert weights[1] > weights[0]


def test_collate_fn(tmp_dataset):
    manifest_path, images_dir, _ = tmp_dataset
    ds = AerialWasteDataset(manifest_path, images_dir)
    items = [ds[i] for i in range(4)]
    batch = AerialWasteDataset.collate_fn(items)
    assert batch["image"].shape == (4, 3, 224, 224)
    assert batch["label"].shape == (4,)
    assert len(batch["source"]) == 4


def test_custom_image_size(tmp_dataset):
    manifest_path, images_dir, _ = tmp_dataset
    ds = AerialWasteDataset(manifest_path, images_dir, image_size=(448, 448))
    item = ds[0]
    assert item["image"].shape == (3, 448, 448)


def test_normalization_applied(tmp_dataset):
    manifest_path, images_dir, _ = tmp_dataset
    norm_stats = {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}
    ds = AerialWasteDataset(manifest_path, images_dir, norm_stats=norm_stats)
    item = ds[0]
    # After normalization, values should go outside [0,1]
    assert item["image"].min() < 0 or item["image"].max() > 1


def test_unknown_task_raises(tmp_dataset):
    manifest_path, images_dir, _ = tmp_dataset
    ds = AerialWasteDataset(manifest_path, images_dir, task="unknown_task")
    with pytest.raises(ValueError, match="Unknown task"):
        _ = ds[0]


def test_segmentation_task_filters_no_mask(tmp_dataset):
    manifest_path, images_dir, _ = tmp_dataset
    # All fixture items have has_mask=False
    ds = AerialWasteDataset(manifest_path, images_dir, task="segmentation")
    assert len(ds) == 0


def test_segmentation_task_with_mask_items(tmp_dataset, tmp_path):
    manifest_path, images_dir, manifest = tmp_dataset
    manifest_with_mask = [dict(m, has_mask=(i == 0)) for i, m in enumerate(manifest)]
    new_manifest = tmp_path / "with_mask.json"
    with open(new_manifest, "w") as f:
        json.dump(manifest_with_mask, f)
    ds = AerialWasteDataset(new_manifest, images_dir, task="segmentation")
    assert len(ds) == 1
    item = ds[0]
    assert item["label"].dtype == torch.long


def test_item_has_id(tmp_dataset):
    manifest_path, images_dir, _ = tmp_dataset
    ds = AerialWasteDataset(manifest_path, images_dir)
    item = ds[0]
    assert "id" in item
    assert isinstance(item["id"], int)


def test_item_has_evidence_and_severity(tmp_dataset):
    manifest_path, images_dir, _ = tmp_dataset
    ds = AerialWasteDataset(manifest_path, images_dir)
    item = ds[0]
    assert "evidence" in item
    assert "severity" in item


def test_source_filter_wv3(tmp_dataset):
    manifest_path, images_dir, _ = tmp_dataset
    ds = AerialWasteDataset(manifest_path, images_dir, source_filter="wv3")
    assert len(ds) == 7
    for item in ds:
        assert item["source"] == "wv3"


def test_source_filter_ge(tmp_dataset):
    manifest_path, images_dir, _ = tmp_dataset
    ds = AerialWasteDataset(manifest_path, images_dir, source_filter="ge")
    assert len(ds) == 6
    for item in ds:
        assert item["source"] == "ge"


def test_source_filter_nonexistent_returns_empty(tmp_dataset):
    manifest_path, images_dir, _ = tmp_dataset
    ds = AerialWasteDataset(manifest_path, images_dir, source_filter="sentinel")
    assert len(ds) == 0


def test_custom_transform_overrides_default_pipeline(tmp_dataset):
    manifest_path, images_dir, _ = tmp_dataset
    from torchvision import transforms as T
    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    ds = AerialWasteDataset(manifest_path, images_dir, transform=transform)
    item = ds[0]
    assert item["image"].shape == (3, 64, 64)


def test_collate_fn_multilabel(tmp_dataset):
    manifest_path, images_dir, _ = tmp_dataset
    ds = AerialWasteDataset(manifest_path, images_dir, task="multilabel")
    items = [ds[i] for i in range(3)]
    batch = AerialWasteDataset.collate_fn(items)
    assert batch["label"].shape == (3, 15)


def test_collate_fn_includes_evidence_and_severity(tmp_dataset):
    manifest_path, images_dir, _ = tmp_dataset
    ds = AerialWasteDataset(manifest_path, images_dir)
    items = [ds[i] for i in range(4)]
    batch = AerialWasteDataset.collate_fn(items)
    assert "evidence" in batch
    assert "severity" in batch
    assert len(batch["evidence"]) == 4
    assert len(batch["severity"]) == 4


def test_collate_fn_ids_are_list(tmp_dataset):
    manifest_path, images_dir, _ = tmp_dataset
    ds = AerialWasteDataset(manifest_path, images_dir)
    items = [ds[i] for i in range(3)]
    batch = AerialWasteDataset.collate_fn(items)
    assert isinstance(batch["id"], list)
    assert len(batch["id"]) == 3


def test_image_tensor_dtype(tmp_dataset):
    manifest_path, images_dir, _ = tmp_dataset
    ds = AerialWasteDataset(manifest_path, images_dir)
    item = ds[0]
    assert item["image"].dtype == torch.float32


def test_image_values_range_without_custom_norm(tmp_dataset):
    """Without custom norm, default ImageNet normalization is applied; values are real floats."""
    manifest_path, images_dir, _ = tmp_dataset
    ds = AerialWasteDataset(manifest_path, images_dir)
    item = ds[0]
    assert item["image"].dtype == torch.float32
    assert item["image"].shape[0] == 3


def test_class_weights_positive_class_higher(tmp_dataset):
    """Minority (positive) class should receive higher weight."""
    manifest_path, images_dir, manifest = tmp_dataset
    pos_count = sum(m["label"] for m in manifest)
    neg_count = len(manifest) - pos_count
    ds = AerialWasteDataset(manifest_path, images_dir)
    weights = ds.class_weights
    if pos_count < neg_count:
        assert weights[1] > weights[0]
    elif neg_count < pos_count:
        assert weights[0] > weights[1]
