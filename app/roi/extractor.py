"""ROI extraction from aligned images.

Crops answer-field regions from an aligned 2100x2970 image using the
RoiField coordinates from a TemplateSpec. Validates bounds and raises
CropOutOfBoundsError on invalid crops.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from app.core.errors import CropOutOfBoundsError, RoiExtractionError
from app.core.schemas import RoiField, TemplateSpec


class CropResult:
    """Result of extracting a single ROI crop.

    Attributes:
        field_id:  the field identifier from RoiField.id
        crop:      numpy array (BGR or grayscale) of the cropped region
        coords:    (x, y, w, h) rectangle used for cropping
        size:      (w, h) of the crop
    """

    def __init__(self, field_id: str, crop: np.ndarray, coords: tuple[int, int, int, int]):
        self.field_id = field_id
        self.crop = crop
        self.coords = coords
        self.size = (coords[2], coords[3])


def extract_single_crop(image: np.ndarray, field: RoiField) -> CropResult:
    """Extract one ROI crop from the aligned image.

    Args:
        image: aligned BGR image (H x W x 3).
        field: RoiField with coords (x, y, w, h).

    Returns:
        CropResult with the cropped numpy array.

    Raises:
        CropOutOfBoundsError: if the crop rectangle exceeds image bounds.
        RoiExtractionError: if OpenCV slicing fails.
    """
    x, y, w, h = field.coords
    img_h, img_w = image.shape[:2]

    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise CropOutOfBoundsError(details={"field_id": field.id, "coords": field.coords, "reason": "negative or zero size"})
    if x + w > img_w or y + h > img_h:
        raise CropOutOfBoundsError(details={"field_id": field.id, "coords": field.coords, "image_size": (img_w, img_h)})

    try:
        crop = image[y:y + h, x:x + w].copy()
    except Exception as e:
        raise RoiExtractionError(details={"field_id": field.id, "error": str(e)})

    if crop.size == 0:
        raise RoiExtractionError(details={"field_id": field.id, "reason": "empty crop"})

    return CropResult(field_id=field.id, crop=crop, coords=field.coords)


def extract_crops(image: np.ndarray, spec: TemplateSpec) -> list[CropResult]:
    """Extract all ROI crops from an aligned image using the template spec.

    Args:
        image: aligned BGR image (H x W x 3).
        spec: TemplateSpec with roi_fields.

    Returns:
        List of CropResult, one per roi_field, in the same order.

    Raises:
        CropOutOfBoundsError: if any crop exceeds image bounds.
    """
    results = []
    for field in spec.roi_fields:
        results.append(extract_single_crop(image, field))
    return results