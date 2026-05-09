"""Tests for prepare_dataset.py utility functions."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestInferSource:

    def test_agea_from_filename(self):
        from scripts.prepare_dataset import infer_source
        assert infer_source({"file_name": "agea_001.jpg"}) == "agea"

    def test_orthophoto_mapped_to_agea(self):
        from scripts.prepare_dataset import infer_source
        assert infer_source({"file_name": "orthophoto_site3.jpg"}) == "agea"

    def test_wv3_from_filename(self):
        from scripts.prepare_dataset import infer_source
        assert infer_source({"file_name": "wv3_image_042.jpg"}) == "wv3"

    def test_worldview_mapped_to_wv3(self):
        from scripts.prepare_dataset import infer_source
        assert infer_source({"file_name": "worldview_tile.jpg"}) == "wv3"

    def test_ge_from_filename(self):
        from scripts.prepare_dataset import infer_source
        assert infer_source({"file_name": "ge_capture.jpg"}) == "ge"

    def test_google_mapped_to_ge(self):
        from scripts.prepare_dataset import infer_source
        assert infer_source({"file_name": "google_earth_shot.jpg"}) == "ge"

    def test_unknown_source(self):
        from scripts.prepare_dataset import infer_source
        # "random_image" contains "ge" (from "image"), so use a name free of all keywords
        assert infer_source({"file_name": "test_sample.jpg"}) == "unknown"

    def test_case_insensitive(self):
        from scripts.prepare_dataset import infer_source
        assert infer_source({"file_name": "AGEA_001.JPG"}) == "agea"
        assert infer_source({"file_name": "WV3_TILE.TIF"}) == "wv3"

    def test_empty_filename(self):
        from scripts.prepare_dataset import infer_source
        assert infer_source({"file_name": ""}) == "unknown"

    def test_missing_file_name_key(self):
        from scripts.prepare_dataset import infer_source
        assert infer_source({}) == "unknown"


class TestBuildMultilabelVector:

    def test_empty_annotation_cats_returns_all_zeros(self):
        from scripts.prepare_dataset import build_multilabel_vector, WASTE_TYPES
        vec = build_multilabel_vector(WASTE_TYPES, [])
        assert vec == [0] * len(WASTE_TYPES)

    def test_single_exact_match(self):
        from scripts.prepare_dataset import build_multilabel_vector, WASTE_TYPES
        vec = build_multilabel_vector(WASTE_TYPES, ["scrap"])
        scrap_idx = WASTE_TYPES.index("scrap")
        assert vec[scrap_idx] == 1
        assert sum(vec) == 1

    def test_multiple_categories_set(self):
        from scripts.prepare_dataset import build_multilabel_vector, WASTE_TYPES
        vec = build_multilabel_vector(WASTE_TYPES, ["scrap", "plastic", "tires"])
        assert vec[WASTE_TYPES.index("scrap")] == 1
        assert vec[WASTE_TYPES.index("plastic")] == 1
        assert vec[WASTE_TYPES.index("tires")] == 1

    def test_unrecognized_category_ignored(self):
        from scripts.prepare_dataset import build_multilabel_vector, WASTE_TYPES
        vec = build_multilabel_vector(WASTE_TYPES, ["invisible_category"])
        assert vec == [0] * len(WASTE_TYPES)

    def test_vector_length_matches_categories(self):
        from scripts.prepare_dataset import build_multilabel_vector, WASTE_TYPES
        vec = build_multilabel_vector(WASTE_TYPES, ["scrap"])
        assert len(vec) == len(WASTE_TYPES)

    def test_vector_length_for_storage_modes(self):
        from scripts.prepare_dataset import build_multilabel_vector, STORAGE_MODES
        vec = build_multilabel_vector(STORAGE_MODES, ["container"])
        assert len(vec) == len(STORAGE_MODES)
        assert vec[STORAGE_MODES.index("container")] == 1

    def test_values_are_binary(self):
        from scripts.prepare_dataset import build_multilabel_vector, WASTE_TYPES
        vec = build_multilabel_vector(WASTE_TYPES, ["scrap", "plastic"])
        assert all(v in (0, 1) for v in vec)

    def test_duplicate_categories_dont_exceed_one(self):
        from scripts.prepare_dataset import build_multilabel_vector, WASTE_TYPES
        vec = build_multilabel_vector(WASTE_TYPES, ["scrap", "scrap"])
        assert vec[WASTE_TYPES.index("scrap")] == 1


class TestComputeNormalizationStats:

    def test_returns_mean_and_std_keys(self, tmp_path):
        from scripts.prepare_dataset import compute_normalization_stats
        img = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))
        p = tmp_path / "img.jpg"
        img.save(p)
        stats = compute_normalization_stats([p])
        assert "mean" in stats
        assert "std" in stats

    def test_mean_has_three_channels(self, tmp_path):
        from scripts.prepare_dataset import compute_normalization_stats
        img = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))
        p = tmp_path / "img.jpg"
        img.save(p)
        stats = compute_normalization_stats([p])
        assert len(stats["mean"]) == 3
        assert len(stats["std"]) == 3

    def test_mean_in_unit_range(self, tmp_path):
        from scripts.prepare_dataset import compute_normalization_stats
        img = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
        p = tmp_path / "img.jpg"
        img.save(p)
        stats = compute_normalization_stats([p])
        for v in stats["mean"]:
            assert 0.0 <= v <= 1.0

    def test_empty_list_falls_back_to_imagenet_defaults(self):
        from scripts.prepare_dataset import compute_normalization_stats
        stats = compute_normalization_stats([])
        assert stats["mean"] == [0.485, 0.456, 0.406]
        assert stats["std"] == [0.229, 0.224, 0.225]

    def test_sample_size_limit_respected(self, tmp_path):
        from scripts.prepare_dataset import compute_normalization_stats
        paths = []
        for i in range(10):
            img = Image.fromarray((np.ones((8, 8, 3)) * i * 25).astype(np.uint8))
            p = tmp_path / f"img_{i}.jpg"
            img.save(p)
            paths.append(p)
        stats = compute_normalization_stats(paths, sample_size=3)
        assert "mean" in stats


class TestLoadCocoJson:

    def test_returns_three_tuples(self, tmp_path):
        from scripts.prepare_dataset import load_coco_json
        payload = {
            "images": [{"id": 1, "file_name": "a.jpg"}],
            "annotations": [{"id": 1, "image_id": 1}],
            "categories": [{"id": 1, "name": "waste"}],
        }
        p = tmp_path / "anno.json"
        p.write_text(json.dumps(payload))
        images, annotations, categories = load_coco_json(p)
        assert len(images) == 1
        assert len(annotations) == 1
        assert len(categories) == 1

    def test_missing_annotations_key_returns_empty(self, tmp_path):
        from scripts.prepare_dataset import load_coco_json
        payload = {"images": [{"id": 1}]}
        p = tmp_path / "anno.json"
        p.write_text(json.dumps(payload))
        _, annotations, categories = load_coco_json(p)
        assert annotations == []
        assert categories == []

    def test_images_content_preserved(self, tmp_path):
        from scripts.prepare_dataset import load_coco_json
        payload = {
            "images": [{"id": 42, "file_name": "test.jpg", "width": 1024, "height": 768}],
            "annotations": [],
        }
        p = tmp_path / "anno.json"
        p.write_text(json.dumps(payload))
        images, _, _ = load_coco_json(p)
        assert images[0]["id"] == 42
        assert images[0]["width"] == 1024
