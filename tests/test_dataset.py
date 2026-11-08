"""Tests for dataset splits and builder (ЛР3)."""

import json
import tempfile
from pathlib import Path

import pytest

from app.dataset.splits import split_photo_ids
from app.dataset.builder import build_dataset
from app.core.schemas import AnswerKey


class TestSplitPhotoIds:
    def test_split_basic(self):
        """Basic split should produce train/val/test with correct sizes."""
        ids = [f"photo_{i}" for i in range(100)]
        result = split_photo_ids(ids, ratios=(0.70, 0.15, 0.15), seed=42)
        assert len(result["train"]) == 70
        assert len(result["val"]) == 15
        assert len(result["test"]) == 15
        # All photo IDs accounted for
        all_ids = result["train"] + result["val"] + result["test"]
        assert set(all_ids) == set(ids)
        assert len(all_ids) == 100

    def test_split_no_leakage(self):
        """No photo_id should appear in more than one split."""
        ids = [f"photo_{i}" for i in range(50)]
        result = split_photo_ids(ids, ratios=(0.70, 0.15, 0.15), seed=123)
        train_set = set(result["train"])
        val_set = set(result["val"])
        test_set = set(result["test"])
        assert train_set & val_set == set()
        assert train_set & test_set == set()
        assert val_set & test_set == set()

    def test_split_reproducible(self):
        """Same seed should produce identical splits."""
        ids = [f"photo_{i}" for i in range(30)]
        r1 = split_photo_ids(ids, seed=42)
        r2 = split_photo_ids(ids, seed=42)
        assert r1 == r2

    def test_split_invalid_ratios(self):
        """Ratios not summing to 1.0 should raise ValueError."""
        with pytest.raises(ValueError):
            split_photo_ids(["a", "b"], ratios=(0.5, 0.3, 0.1))


class TestBuildDataset:
    def test_build_dataset_basic(self, tmp_path):
        """build_dataset should create crops/, labels.csv, and manifest."""
        # Create a fake crop
        crops_dir = tmp_path / "crops_in"
        crops_dir.mkdir()
        import cv2
        import numpy as np
        img = np.ones((50, 100, 3), dtype=np.uint8) * 200
        cv2.imwrite(str(crops_dir / "q1_answer.png"), img)

        out_dir = tmp_path / "dataset_out"
        manifest = build_dataset(
            crops_dir=crops_dir,
            labels_path=tmp_path / "nonexistent.csv",
            output_dir=out_dir,
            template_version="v1",
            photo_id="photo_001",
            split="train",
        )

        assert (out_dir / "crops").exists()
        assert (out_dir / "labels.csv").exists()
        assert (out_dir / "dataset_manifest.json").exists()
        assert manifest["template_version"] == "v1"
        assert manifest["photo_id"] == "photo_001"
        assert manifest["split"] == "train"
        assert manifest["total_crops"] >= 1

    def test_build_dataset_with_answer_key(self, tmp_path):
        """AnswerKey cross-check should populate labels."""
        crops_dir = tmp_path / "crops_in"
        crops_dir.mkdir()
        import cv2
        import numpy as np
        img = np.ones((50, 100, 3), dtype=np.uint8) * 200
        cv2.imwrite(str(crops_dir / "q1_answer.png"), img)

        ak = AnswerKey(template_version="v1", answers={"q1_answer": "A"})
        out_dir = tmp_path / "dataset_out"
        manifest = build_dataset(
            crops_dir=crops_dir,
            labels_path=tmp_path / "nonexistent.csv",
            output_dir=out_dir,
            template_version="v1",
            photo_id="photo_002",
            answer_key=ak,
        )
        # labels.csv should contain the answer key label
        labels_csv = (out_dir / "labels.csv").read_text(encoding="utf-8")
        assert "A" in labels_csv

    def test_build_dataset_answer_key_mismatch(self, tmp_path):
        """Mismatched template_version in AnswerKey should raise."""
        ak = AnswerKey(template_version="v2", answers={})
        with pytest.raises(Exception):
            build_dataset(
                crops_dir=tmp_path,
                labels_path=tmp_path / "x.csv",
                output_dir=tmp_path / "out",
                template_version="v1",
                photo_id="p1",
                answer_key=ak,
            )