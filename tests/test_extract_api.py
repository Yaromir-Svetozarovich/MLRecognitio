"""Tests for the /api/extract route (ЛР3 — полезная нагрузка / ROI-поля).

Проверяет: успех, неверный шаблон, слишком большой файл, невалидное изображение.
"""

import io
import json

import cv2
import numpy as np
import pytest

from fastapi.testclient import TestClient

from app.api.main import app


def _make_aligned_photo() -> bytes:
    """Create a synthetic aligned 2100x2970 image with a dark square in ROI area."""
    img = np.ones((2970, 2100, 3), dtype=np.uint8) * 255
    # dark rectangle at (1390, 780, 560, 70) — matches blank_v1 q1_answer ROI
    img[780:850, 1390:1950] = (50, 50, 50)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _make_spec_file(tmp_path) -> str:
    """Write a minimal TemplateSpec with one ROI field and return its path."""
    spec = {
        "template_version": "test_extract_v1",
        "canvas_size": [2100, 2970],
        "roi_fields": [
            {"id": "q1_answer", "type": "text", "coords": [1390, 780, 560, 70]}
        ],
        "corner_by_id": {"0": "TL", "1": "TR", "2": "BR", "3": "BL"},
        "expected_marker_ids": [0, 1, 2, 3],
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return str(spec_path)


class TestExtractApi:
    def test_extract_success(self, tmp_path):
        """Valid aligned image + spec -> 200 with one crop as data URL."""
        client = TestClient(app)
        spec_path = _make_spec_file(tmp_path)
        photo_bytes = _make_aligned_photo()

        r = client.post(
            "/api/extract",
            files={"photo": ("aligned.jpg", photo_bytes, "image/jpeg")},
            data={"template_id": spec_path},
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["status"] == "ok"
        assert data["template_version"] == "test_extract_v1"
        assert data["num_crops"] == 1
        crop = data["crops"][0]
        assert crop["field_id"] == "q1_answer"
        assert crop["coords"] == [1390, 780, 560, 70]
        assert crop["size"] == [560, 70]
        assert crop["crop_image"].startswith("data:image/png;base64,")
        assert len(crop["crop_image"]) > 100

    def test_extract_missing_template_id(self, tmp_path):
        """No template_id -> 422 TemplateSpecRequired."""
        client = TestClient(app)
        photo_bytes = _make_aligned_photo()
        r = client.post(
            "/api/extract",
            files={"photo": ("aligned.jpg", photo_bytes, "image/jpeg")},
        )
        assert r.status_code == 422
        assert r.json()["detail"]["code"] == "TemplateSpecRequired"

    def test_extract_invalid_template_id(self, tmp_path):
        """Non-existent template_id -> 404 JobNotFoundError."""
        client = TestClient(app)
        photo_bytes = _make_aligned_photo()
        r = client.post(
            "/api/extract",
            files={"photo": ("aligned.jpg", photo_bytes, "image/jpeg")},
            data={"template_id": "/nonexistent/spec.json"},
        )
        assert r.status_code == 404
        assert r.json()["detail"]["code"] == "JobNotFoundError"

    def test_extract_invalid_image(self, tmp_path):
        """Non-image upload -> 400 InvalidImageError."""
        client = TestClient(app)
        spec_path = _make_spec_file(tmp_path)
        r = client.post(
            "/api/extract",
            files={"photo": ("not_an_image.txt", b"hello world", "text/plain")},
            data={"template_id": spec_path},
        )
        assert r.status_code == 400
        assert r.json()["detail"]["code"] == "InvalidImageError"

    def test_extract_too_large(self, tmp_path, monkeypatch):
        """Oversized upload -> 413 UploadTooLargeError."""
        from app.core.config import AppConfig

        client = TestClient(app)
        spec_path = _make_spec_file(tmp_path)

        # Temporarily lower max_upload_mb to 0 so any file is "too large"
        orig_from_env = AppConfig.from_env

        def fake_from_env():
            cfg = orig_from_env()
            cfg.max_upload_mb = 0
            return cfg

        monkeypatch.setattr("app.api.routes.extract.get_config", fake_from_env)

        photo_bytes = _make_aligned_photo()
        r = client.post(
            "/api/extract",
            files={"photo": ("aligned.jpg", photo_bytes, "image/jpeg")},
            data={"template_id": spec_path},
        )
        assert r.status_code == 413
        assert r.json()["detail"]["code"] == "UploadTooLargeError"
