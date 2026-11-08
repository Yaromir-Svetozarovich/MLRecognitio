"""Tests for ROI extraction (ЛР3)."""

import cv2
import numpy as np
import pytest

from app.core.schemas import RoiField, TemplateSpec
from app.roi.extractor import CropOutOfBoundsError, extract_crops, extract_single_crop
from app.roi.spec_loader import validate_roi_fields


def _make_aligned_image(w: int = 2100, h: int = 2970) -> np.ndarray:
    """Create a blank white aligned image."""
    return np.ones((h, w, 3), dtype=np.uint8) * 255


def _make_spec_with_rois() -> TemplateSpec:
    """Create a TemplateSpec with two ROI fields inside the canvas."""
    return TemplateSpec(
        template_version="test_v1",
        canvas_size=(2100, 2970),
        roi_fields=[
            RoiField(id="q1_answer", type="text", coords=(100, 100, 200, 50)),
            RoiField(id="q2_answer", type="digit", coords=(100, 200, 100, 30)),
        ],
    )


class TestExtractSingleCrop:
    def test_extract_valid_crop(self):
        """A valid ROI should produce a crop of the correct size."""
        image = _make_aligned_image()
        field = RoiField(id="field1", type="text", coords=(50, 50, 100, 40))
        result = extract_single_crop(image, field)
        assert result.field_id == "field1"
        assert result.size == (100, 40)
        assert result.crop.shape == (40, 100, 3)

    def test_extract_out_of_bounds(self):
        """ROI exceeding image bounds should raise CropOutOfBoundsError."""
        image = _make_aligned_image()
        field = RoiField(id="field1", type="text", coords=(2000, 2900, 200, 100))
        with pytest.raises(CropOutOfBoundsError):
            extract_single_crop(image, field)

    def test_extract_negative_coords(self):
        """Negative coords should raise CropOutOfBoundsError."""
        image = _make_aligned_image()
        field = RoiField(id="field1", type="text", coords=(-10, 50, 100, 40))
        with pytest.raises(CropOutOfBoundsError):
            extract_single_crop(image, field)


class TestExtractCrops:
    def test_extract_all_crops(self):
        """extract_crops should return one CropResult per roi_field."""
        image = _make_aligned_image()
        spec = _make_spec_with_rois()
        crops = extract_crops(image, spec)
        assert len(crops) == 2
        assert crops[0].field_id == "q1_answer"
        assert crops[1].field_id == "q2_answer"

    def test_extract_empty_spec(self):
        """A spec with no roi_fields should return an empty list."""
        image = _make_aligned_image()
        spec = TemplateSpec(template_version="empty", canvas_size=(2100, 2970), roi_fields=[])
        crops = extract_crops(image, spec)
        assert crops == []


class TestValidateRoiFields:
    def test_valid_rois(self):
        """All ROIs within canvas should produce no errors."""
        spec = _make_spec_with_rois()
        errors = validate_roi_fields(spec)
        assert errors == []

    def test_out_of_bounds_roi(self):
        """ROI exceeding canvas should produce an error string."""
        spec = TemplateSpec(
            template_version="test",
            canvas_size=(2100, 2970),
            roi_fields=[RoiField(id="bad", type="text", coords=(2000, 2800, 200, 200))],
        )
        errors = validate_roi_fields(spec)
        assert len(errors) == 1
        assert "bad" in errors[0]