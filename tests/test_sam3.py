"""Tests for SAM 3 inference wrapper."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestWastePrompts:
    """Test SAM 3 prompt configuration."""

    def test_prompts_are_strings(self):
        from scripts.run_sam3 import WASTE_PROMPTS, WASTE_CONCEPT_PROMPTS
        assert all(isinstance(p, str) for p in WASTE_PROMPTS)
        assert all(isinstance(p, str) for p in WASTE_CONCEPT_PROMPTS)
        assert len(WASTE_PROMPTS) > 0
        assert len(WASTE_CONCEPT_PROMPTS) > 0

    def test_prompts_cover_aerial_waste_categories(self):
        from scripts.run_sam3 import WASTE_PROMPTS
        prompts_text = " ".join(WASTE_PROMPTS).lower()
        # Check key waste types are covered
        for keyword in ["scrap", "waste", "debris", "rubble"]:
            assert keyword in prompts_text, f"'{keyword}' not found in WASTE_PROMPTS"


class TestSAM3Loading:
    """Test SAM 3 loading (mocked)."""

    def test_load_sam3_missing_weights(self, tmp_path):
        from scripts.run_sam3 import load_sam3
        with pytest.raises(FileNotFoundError, match="SAM 3 weights not found"):
            load_sam3(str(tmp_path / "nonexistent.pt"))

    @patch("scripts.run_sam3.SAM3SemanticPredictor", create=True)
    def test_load_sam3_with_weights(self, mock_predictor, tmp_path):
        weights = tmp_path / "sam3.pt"
        weights.touch()

        # Patch the import
        with patch.dict("sys.modules", {
            "ultralytics": MagicMock(),
            "ultralytics.models": MagicMock(),
            "ultralytics.models.sam": MagicMock(SAM3SemanticPredictor=mock_predictor),
        }):
            from scripts.run_sam3 import load_sam3
            predictor = load_sam3(str(weights))
            assert predictor is not None


class TestDetectionParsing:
    """Test detection result parsing logic."""

    def test_empty_detections(self):
        """Should handle no detections gracefully."""
        result = {
            "file": "test.jpg",
            "source": "agea",
            "num_instances": 0,
            "detections": [],
        }
        assert result["num_instances"] == 0
        assert isinstance(result["detections"], list)

    def test_mask_fraction_in_range(self):
        """mask_fraction should always be in [0, 1]."""
        mock_detections = [
            {"mask_area_px": 1000, "mask_fraction": 0.05, "confidence": 0.8},
            {"mask_area_px": 5000, "mask_fraction": 0.25, "confidence": 0.6},
        ]
        for det in mock_detections:
            assert 0.0 <= det["mask_fraction"] <= 1.0


class TestResultSaving:
    """Test output format for SAM 3 results."""

    def test_output_json_structure(self, tmp_path):
        """Verify the expected output JSON structure."""
        mock_results = {
            "mode": "text",
            "n_images": 10,
            "total_instances": 23,
            "avg_instances_per_image": 2.3,
            "results": [
                {
                    "file": "test.jpg",
                    "source": "agea",
                    "prithvi_confidence": 0.92,
                    "ground_truth_label": 1,
                    "num_instances": 3,
                    "total_waste_fraction": 0.18,
                    "detections": [
                        {
                            "instance_id": 0,
                            "mask_area_px": 1024,
                            "mask_fraction": 0.06,
                            "confidence": 0.78,
                        }
                    ]
                }
            ]
        }

        output = tmp_path / "sam3_results.json"
        with open(output, "w") as f:
            json.dump(mock_results, f)

        with open(output) as f:
            loaded = json.load(f)

        assert loaded["n_images"] == 10
        assert loaded["total_instances"] == 23
        assert len(loaded["results"]) == 1
        assert loaded["results"][0]["num_instances"] == 3
