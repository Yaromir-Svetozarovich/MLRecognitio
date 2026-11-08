"""End-to-end tests for the normalizer (ЛР2).

Синтетические фото с 4 ArUco-маркерами имитируют реальную фотографию
распечатанного бланка, сделанную под углом. Проверяется полный путь:
сервис -> предобработка -> поиск маркеров -> гомография -> варп.

Запуск: python -m pytest tests/test_align_e2e.py -q
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.config import AppConfig
from app.core.schemas import AlignConfig, TemplateSpec
from app.normalizer import aligner, markers, preprocessing
from app.normalizer.service import NormalizerService


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_synthetic_photo(markers: dict[int, tuple[int, int]],
                         photo_size=(1600, 1200)) -> np.ndarray:
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
        img = cv2.copyMakeBorder(img, quiet, quiet, quiet, quiet,
                                 cv2.BORDER_CONSTANT, value=255)
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


def load_real_spec() -> TemplateSpec:
    """Load the real TemplateSpec from output/blank_v1.spec.json if present."""
    spec_path = Path("output/blank_v1.spec.json")
    if not spec_path.exists():
        pytest.skip("output/blank_v1.spec.json not found — generate it first")
    data = json.loads(spec_path.read_text(encoding="utf-8"))
    return TemplateSpec.model_validate(data)


MARKER_POSITIONS = {0: (150, 100), 1: (1450, 100),
                    2: (1450, 1100), 3: (150, 1100)}


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------

class TestPreprocessingE2E:
    def test_preprocess_synthetic_photo(self):
        photo = make_synthetic_photo(MARKER_POSITIONS)
        gray, edges = preprocessing.preprocess(photo, (50, 150))
        assert gray.shape == photo.shape[:2]
        assert gray.dtype == np.uint8
        assert edges.shape == photo.shape[:2]
        assert edges.dtype == np.uint8


# ---------------------------------------------------------------------------
# markers
# ---------------------------------------------------------------------------

class TestMarkersE2E:
    def test_detect_all_four_markers(self):
        photo = make_synthetic_photo(MARKER_POSITIONS)
        corners, ids = markers.detect_markers(photo, cv2.aruco.DICT_4X4_50)
        assert ids is not None
        flat = ids.flatten().tolist()
        assert set(flat) == {0, 1, 2, 3}

    def test_validate_all_four_ok(self):
        photo = make_synthetic_photo(MARKER_POSITIONS)
        corners, ids = markers.detect_markers(photo, cv2.aruco.DICT_4X4_50)
        flat = markers.validate_markers(ids, [0, 1, 2, 3])
        assert set(flat) == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# aligner
# ---------------------------------------------------------------------------

class TestAlignerE2E:
    def _make_spec(self) -> TemplateSpec:
        return TemplateSpec(
            template_version="test",
            canvas_size=(2100, 2970),
            marker_layout={0: (175, 175), 1: (1925, 175),
                           2: (1925, 2795), 3: (175, 2795)},
            marker_size_px=150,
        )

    def test_align_centers_mode(self):
        spec = self._make_spec()
        config = AlignConfig(corner_mode="centers")
        photo = make_synthetic_photo(MARKER_POSITIONS)
        output_image, edges_image, result = aligner.align(photo, config, spec)
        assert output_image.shape == (2970, 2100, 3)
        assert edges_image is not None
        assert result.corner_mode == "centers"
        assert set(result.marker_ids) == {0, 1, 2, 3}

    def test_align_centers_to_layout_mode(self):
        spec = self._make_spec()
        config = AlignConfig(corner_mode="centers_to_layout")
        photo = make_synthetic_photo(MARKER_POSITIONS)
        output_image, edges_image, result = aligner.align(photo, config, spec)
        assert output_image.shape == (2970, 2100, 3)
        assert result.corner_mode == "centers_to_layout"
        assert set(result.marker_ids) == {0, 1, 2, 3}

    def test_align_real_spec_from_file(self):
        """Alignment against the real blank_v1 spec (canonical mapping + ROI geometry)."""
        spec = load_real_spec()
        config = AlignConfig(corner_mode="centers_to_layout")
        photo = make_synthetic_photo(MARKER_POSITIONS)
        output_image, edges_image, result = aligner.align(photo, config, spec)
        assert output_image.shape == (2970, 2100, 3)
        assert set(result.marker_ids) == set(spec.expected_marker_ids)
        assert result.corner_mode == "centers_to_layout"

    def test_align_missing_marker_raises(self):
        """Only 3 markers -> MarkersIncompleteError."""
        three = {k: v for k, v in MARKER_POSITIONS.items() if k != 3}
        photo = make_synthetic_photo(three)
        spec = self._make_spec()
        config = AlignConfig(corner_mode="centers")
        with pytest.raises(Exception):
            aligner.align(photo, config, spec)


# ---------------------------------------------------------------------------
# service (the layer the API/CLI call)
# ---------------------------------------------------------------------------

class TestNormalizerServiceE2E:
    def test_align_accepts_raw_bytes_like_api(self, tmp_path):
        """The API passes uploaded bytes; the service must decode them."""
        photo = make_synthetic_photo(MARKER_POSITIONS)
        ok, buf = cv2.imencode(".jpg", photo)
        assert ok

        service = NormalizerService(AppConfig(data_dir=tmp_path))
        output_image, edges_image, result = service.align(buf.tobytes())

        assert output_image.shape == (2970, 2100, 3)
        assert edges_image is not None
        assert set(result.marker_ids) == {0, 1, 2, 3}

    def test_align_accepts_file_path(self, tmp_path):
        photo = make_synthetic_photo(MARKER_POSITIONS)
        photo_path = tmp_path / "photo.jpg"
        cv2.imwrite(str(photo_path), photo)

        config = AppConfig(data_dir=tmp_path)
        config.normalizer.save_edges = True
        service = NormalizerService(config)
        output_image, edges_image, result = service.align_file(
            str(photo_path),
            save_aligned_path=str(tmp_path / "aligned.png"),
            save_edges_path=str(tmp_path / "edges.png"),
        )
        assert output_image.shape == (2970, 2100, 3)
        assert (tmp_path / "aligned.png").is_file()
        assert (tmp_path / "edges.png").is_file()
        assert set(result.marker_ids) == {0, 1, 2, 3}

    def test_align_with_real_spec_bytes(self, tmp_path):
        """Service align with bytes + real TemplateSpec (centers_to_layout)."""
        spec = load_real_spec()
        photo = make_synthetic_photo(MARKER_POSITIONS)
        ok, buf = cv2.imencode(".png", photo)
        assert ok

        service = NormalizerService(AppConfig(data_dir=tmp_path))
        output_image, edges_image, result = service.align(
            buf.tobytes(),
            template_spec=spec,
            corner_mode="centers_to_layout",
        )
        assert output_image.shape == (2970, 2100, 3)
        assert set(result.marker_ids) == set(spec.expected_marker_ids)


# ---------------------------------------------------------------------------
# API-level sanity (mirrors what POST /api/align does)
# ---------------------------------------------------------------------------

class TestAPIAlignmentSanity:
    def test_align_end_to_end_via_service_mimics_api(self, tmp_path):
        """Simulates API flow: raw bytes -> service -> aligned 2100x2970."""
        spec = load_real_spec()
        photo = make_synthetic_photo(MARKER_POSITIONS)
        ok, buf = cv2.imencode(".jpg", photo)
        assert ok

        service = NormalizerService(AppConfig(data_dir=tmp_path))
        output_image, edges_image, result = service.align(
            buf.tobytes(),
            template_spec=spec,
            corner_mode="centers_to_layout",
        )

        # API contract checks
        assert output_image.shape == (2970, 2100, 3)   # h x w x 3
        assert list(result.output_size) == [2100, 2970]
        assert result.corner_mode == "centers_to_layout"
        assert sorted(result.marker_ids) == sorted(spec.expected_marker_ids)
        assert edges_image is not None
