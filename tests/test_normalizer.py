"""Tests for the normalizer (ЛР2).

Covers preprocessing, marker detection/validation, and alignment (all 3 corner_mode).
"""

import cv2
import numpy as np
import pytest

from app.core.schemas import AlignConfig, TemplateSpec
from app.core.config import AppConfig
from app.normalizer import aligner, markers, preprocessing
from app.normalizer.service import NormalizerService


def make_synthetic_photo(markers: dict[int, tuple[int, int]], photo_size=(1600, 1200)) -> np.ndarray:
    """Create a synthetic photo with 4 ArUco markers at given pixel positions.

    Draws markers on a white canvas, then applies a mild perspective warp
    to simulate a real photo of a printed sheet taken at an angle.
    """
    W, H = photo_size
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size = 100
    quiet = 20
    total = marker_size + 2 * quiet
    for mid, (cx, cy) in markers.items():
        img = cv2.aruco.generateImageMarker(d, mid, marker_size)
        img = cv2.copyMakeBorder(img, quiet, quiet, quiet, quiet, cv2.BORDER_CONSTANT, value=255)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        x0 = int(cx - total // 2)
        y0 = int(cy - total // 2)
        x0 = max(0, min(x0, W - total))
        y0 = max(0, min(y0, H - total))
        canvas[y0:y0 + total, x0:x0 + total] = img

    src_pts = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    dst_pts = np.float32([[30, 30], [W - 50, 20], [W - 40, H - 60], [20, H - 50]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    photo = cv2.warpPerspective(canvas, M, (W, H), borderValue=(200, 200, 200))
    return photo


MARKER_POSITIONS = {0: (150, 100), 1: (1450, 100), 2: (1450, 1100), 3: (150, 1100)}


class TestPreprocessing:
    def test_grayscale_and_canny(self):
        """Grayscale and Canny should produce correct-shaped uint8 outputs."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:] = (100, 150, 200)
        gray, edges = preprocessing.preprocess(img, (50, 150))
        assert gray.shape == (100, 100)
        assert edges.shape == (100, 100)
        assert edges.dtype == np.uint8


class TestMarkers:
    def test_detect_and_validate_ok(self):
        """Detecting 4 markers on a synthetic photo should return all IDs."""
        photo = make_synthetic_photo(MARKER_POSITIONS)
        corners, ids = markers.detect_markers(photo, cv2.aruco.DICT_4X4_50)
        flat = markers.validate_markers(ids, [0, 1, 2, 3])
        assert set(flat) == {0, 1, 2, 3}

    def test_validate_markers_not_found(self):
        """None ids should raise MarkersNotFoundError."""
        with pytest.raises(Exception):
            markers.validate_markers(None, [0, 1, 2, 3])

    def test_validate_markers_incomplete(self):
        """Only 3 marker IDs should raise MarkersIncompleteError."""
        ids = np.array([[0], [1], [2]], dtype=np.int32)
        with pytest.raises(Exception):
            markers.validate_markers(ids, [0, 1, 2, 3])


class TestAligner:
    def _make_spec(self):
        return TemplateSpec(
            template_version="test",
            canvas_size=(2100, 2970),
            marker_layout={0: (175, 175), 1: (1925, 175), 2: (1925, 2795), 3: (175, 2795)},
            marker_size_px=150,
        )

    def test_align_centers_to_layout(self):
        """centers_to_layout mode should map marker centres to marker_layout positions."""
        spec = self._make_spec()
        config = AlignConfig(corner_mode="centers_to_layout")
        photo = make_synthetic_photo(MARKER_POSITIONS)

        output_image, edges, result = aligner.align(photo, config, spec)
        assert output_image.shape == (2970, 2100, 3)
        assert result.output_size == (2100, 2970)
        assert result.corner_mode == "centers_to_layout"
        assert set(result.marker_ids) == {0, 1, 2, 3}

    def test_align_centers(self):
        """centers mode should map marker centres to canvas corners."""
        spec = self._make_spec()
        config = AlignConfig(corner_mode="centers")
        photo = make_synthetic_photo(MARKER_POSITIONS)

        output_image, edges, result = aligner.align(photo, config, spec)
        assert output_image.shape == (2970, 2100, 3)
        assert result.corner_mode == "centers"

    def test_align_centers_to_layout_without_spec_raises(self):
        """centers_to_layout without TemplateSpec should raise ConfigError."""
        config = AlignConfig(corner_mode="centers_to_layout")
        photo = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
        with pytest.raises(Exception):
            aligner.align(photo, config, None)


class TestNormalizerService:
    def test_align_accepts_raw_bytes(self, tmp_path):
        """The API passes uploaded bytes; the service must decode them."""
        photo = make_synthetic_photo(MARKER_POSITIONS)
        ok, buf = cv2.imencode(".jpg", photo)
        assert ok

        service = NormalizerService(AppConfig(data_dir=tmp_path))
        output_image, edges_image, result = service.align(buf.tobytes())

        assert output_image.shape == (2970, 2100, 3)
        assert edges_image is not None
        assert set(result.marker_ids) == {0, 1, 2, 3}