"""Tests for evaluate.py utility functions (pure logic, no file I/O to results/)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSortByDisplayOrder:

    def test_known_models_maintain_canonical_order(self):
        from scripts.evaluate import _sort_by_display_order
        unsorted = {
            "Prithvi-300M + LoRA": {},
            "ResNet50+FPN (paper)": {},
            "ResNet50+FPN (reproduced)": {},
        }
        result = _sort_by_display_order(unsorted)
        keys = list(result.keys())
        assert keys.index("ResNet50+FPN (paper)") < keys.index("ResNet50+FPN (reproduced)")
        assert keys.index("ResNet50+FPN (reproduced)") < keys.index("Prithvi-300M + LoRA")

    def test_unknown_models_sorted_after_known(self):
        from scripts.evaluate import _sort_by_display_order
        d = {"Unknown Model": {}, "ResNet50+FPN (paper)": {}}
        result = _sort_by_display_order(d)
        keys = list(result.keys())
        assert keys[-1] == "Unknown Model"

    def test_empty_dict_returns_empty(self):
        from scripts.evaluate import _sort_by_display_order
        assert _sort_by_display_order({}) == {}

    def test_single_entry_unchanged(self):
        from scripts.evaluate import _sort_by_display_order
        d = {"ResNet50+FPN (paper)": {"AP": 0.88}}
        result = _sort_by_display_order(d)
        assert list(result.keys()) == ["ResNet50+FPN (paper)"]


class TestOptimalThreshold:

    def test_threshold_in_unit_range(self):
        from scripts.evaluate import _optimal_threshold
        np.random.seed(0)
        y_true = np.random.randint(0, 2, 50).tolist()
        y_prob = np.random.rand(50).tolist()
        threshold = _optimal_threshold(y_true, y_prob)
        assert 0.0 <= threshold <= 1.0

    def test_perfect_separator_threshold_between_classes(self):
        from scripts.evaluate import _optimal_threshold
        y_true = [0, 0, 1, 1]
        y_prob = [0.1, 0.2, 0.8, 0.9]
        threshold = _optimal_threshold(y_true, y_prob)
        assert 0.2 <= threshold <= 0.8

    def test_returns_float(self):
        from scripts.evaluate import _optimal_threshold
        threshold = _optimal_threshold([0, 1], [0.3, 0.7])
        assert isinstance(threshold, float)


class TestLoadAllResults:

    def test_loads_known_run_name(self, tmp_path):
        from scripts.evaluate import load_all_results
        payload = {
            "run_name": "resnet50_fpn_all",
            "test_metrics": {"AP": 0.87},
            "param_info": {"trainable_pct": 100.0},
        }
        (tmp_path / "resnet50_fpn_all_results.json").write_text(json.dumps(payload))
        loaded = load_all_results(tmp_path)
        assert "ResNet50+FPN (reproduced)" in loaded

    def test_empty_directory_returns_empty_dict(self, tmp_path):
        from scripts.evaluate import load_all_results
        assert load_all_results(tmp_path) == {}

    def test_run_name_fallback_uses_filename_stem(self, tmp_path):
        from scripts.evaluate import load_all_results
        payload = {"test_metrics": {"AP": 0.7}}
        (tmp_path / "custom_model_results.json").write_text(json.dumps(payload))
        loaded = load_all_results(tmp_path)
        assert "custom_model" in loaded

    def test_multiple_files_all_loaded(self, tmp_path):
        from scripts.evaluate import load_all_results
        for name in ("resnet50_fpn_all", "prithvi_300m_lora_all"):
            payload = {"run_name": name, "test_metrics": {}, "param_info": {}}
            (tmp_path / f"{name}_results.json").write_text(json.dumps(payload))
        loaded = load_all_results(tmp_path)
        assert len(loaded) == 2


class TestLoadAllPredictions:

    def test_loads_predictions_for_known_model(self, tmp_path):
        from scripts.evaluate import load_all_predictions
        payload = {
            "model": "resnet50_fpn_all",
            "predictions": [
                {"source": "agea", "prob_positive": 0.9, "label": 1},
                {"source": "agea", "prob_positive": 0.1, "label": 0},
            ],
        }
        (tmp_path / "resnet50_fpn_all_test_preds.json").write_text(json.dumps(payload))
        loaded = load_all_predictions(tmp_path)
        assert "ResNet50+FPN (reproduced)" in loaded
        assert len(loaded["ResNet50+FPN (reproduced)"]) == 2

    def test_empty_directory_returns_empty_dict(self, tmp_path):
        from scripts.evaluate import load_all_predictions
        assert load_all_predictions(tmp_path) == {}

    def test_model_field_fallback_to_filename(self, tmp_path):
        from scripts.evaluate import load_all_predictions
        payload = {"predictions": [{"prob_positive": 0.5, "label": 1}]}
        (tmp_path / "my_model_test_preds.json").write_text(json.dumps(payload))
        loaded = load_all_predictions(tmp_path)
        assert "my_model" in loaded


class TestBuildComparisonTable:

    def test_includes_paper_reference_row(self):
        from scripts.evaluate import build_comparison_table
        df = build_comparison_table({})
        assert "ResNet50+FPN (paper)" in df["Model"].values

    def test_paper_ap_value_matches_reported(self):
        from scripts.evaluate import build_comparison_table
        df = build_comparison_table({})
        row = df[df["Model"] == "ResNet50+FPN (paper)"].iloc[0]
        assert abs(row["AP (all)"] - 0.8799) < 1e-4

    def test_custom_results_included_in_table(self):
        from scripts.evaluate import build_comparison_table
        custom = {
            "ResNet50+FPN (reproduced)": {
                "test_metrics": {"AP": 0.85, "F1": 0.78, "precision": 0.80, "recall": 0.76},
                "param_info": {"trainable_pct": 80.0},
            }
        }
        df = build_comparison_table(custom)
        assert "ResNet50+FPN (reproduced)" in df["Model"].values
        row = df[df["Model"] == "ResNet50+FPN (reproduced)"].iloc[0]
        assert abs(row["AP (all)"] - 0.85) < 1e-4

    def test_paper_row_comes_before_custom_rows(self):
        from scripts.evaluate import build_comparison_table
        custom = {
            "Prithvi-300M + LoRA": {
                "test_metrics": {"AP": 0.91, "F1": 0.85},
                "param_info": {"trainable_pct": 1.6},
            }
        }
        df = build_comparison_table(custom)
        models = list(df["Model"])
        paper_idx = models.index("ResNet50+FPN (paper)")
        lora_idx  = models.index("Prithvi-300M + LoRA")
        assert paper_idx < lora_idx

    def test_missing_metrics_do_not_raise(self):
        from scripts.evaluate import build_comparison_table
        custom = {"ResNet50+FPN (reproduced)": {}}
        df = build_comparison_table(custom)
        assert df is not None
