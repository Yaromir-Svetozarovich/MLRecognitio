"""Extract route: aligned image -> ROI crops (useful payload).

POST /api/extract accepts multipart/form-data with an aligned photo file and
optional template_id. Returns JSON with base64-encoded crop data URLs for
each answer field (q1_answer, q2_answer, ...) plus coordinates.
"""

from __future__ import annotations

import base64
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from app.api.deps import get_config
from app.core.errors import DomainError
from app.core.schemas import TemplateSpec
from app.core.storage import LocalStorage
from app.normalizer.service import NormalizerService
from app.roi.extractor import CropOutOfBoundsError, extract_crops
from app.roi.spec_loader import load_template_spec

router = APIRouter()


def _encode_png(image) -> bytes:
    """Encode a numpy BGR/grayscale image as PNG bytes."""
    import cv2

    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Failed to encode image as PNG")
    return buf.tobytes()


def _to_data_url(png_bytes: bytes) -> str:
    """Wrap PNG bytes in a base64 data URL for direct use in <img src>."""
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


@router.post("")
def extract(
    photo: UploadFile = File(...),
    template_id: str = Form(None),
):
    """Extract ROI crops (useful payload) from an aligned image.

    Steps:
      1. Reject uploads larger than max_upload_mb before reading into memory.
      2. Read raw bytes from UploadFile and decode to numpy array.
      3. Resolve TemplateSpec (from template_id or config defaults).
      4. Call extract_crops() -> list[CropResult] with per-field crops.
      5. Encode each crop as base64 data URL for immediate Web preview.
      6. Return CropBatch JSON with field_id, coords, size, and data URL.
    """
    config = get_config()

    # Size check before reading into memory (protects against large uploads)
    if photo.size and photo.size > config.max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail={"code": "UploadTooLargeError", "message": "File too large"},
        )

    raw = photo.file.read()

    # Decode uploaded bytes to numpy array (same as align does via read_image)
    import cv2
    import numpy as np

    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail={"code": "InvalidImageError", "message": "Cannot decode uploaded image"},
        )

    # Resolve TemplateSpec
    try:
        template_spec = load_template_spec(template_id, config) if template_id else None
    except DomainError as e:
        raise HTTPException(
            status_code=e.http_status,
            detail={"code": e.code, "message": e.message, "details": e.details},
        )
    if template_spec is None:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "TemplateSpecRequired",
                "message": "template_id is required for extraction",
                "details": {"hint": "Provide ?template_id= or form field template_id"},
            },
        )

    # Validate ROI fields fit within canvas before extracting
    from app.roi.spec_loader import validate_roi_fields

    roi_errors = validate_roi_fields(template_spec)
    if roi_errors:
        raise HTTPException(
            status_code=422,
            detail={"code": "InvalidRoiFields", "message": "; ".join(roi_errors)},
        )

    try:
        results = extract_crops(image, template_spec)
    except CropOutOfBoundsError as e:
        raise HTTPException(
            status_code=400,
            detail={"code": e.code, "message": e.message, "details": e.details},
        )
    except DomainError as e:
        raise HTTPException(
            status_code=e.http_status,
            detail={"code": e.code, "message": e.message, "details": e.details},
        )

    # Encode each crop as base64 data URL
    crops_out = []
    for crop in results:
        crop_bytes = _encode_png(crop.crop)
        crops_out.append({
            "field_id": crop.field_id,
            "coords": list(crop.coords),
            "size": list(crop.size),
            "crop_image": _to_data_url(crop_bytes),
        })

    # Optionally save crops to disk
    sc = config.normalizer
    if getattr(sc, "save_crops", False):
        storage = LocalStorage(config.data_dir)
        for crop in results:
            storage.save_bytes(
                f"jobs/extract/{template_spec.template_version}/{crop.field_id}.png",
                _encode_png(crop.crop),
            )

    return JSONResponse({
        "status": "ok",
        "template_version": template_spec.template_version,
        "num_crops": len(crops_out),
        "crops": crops_out,
    })
