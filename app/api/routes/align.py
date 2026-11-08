"""Align route: photo -> aligned_blank.png + edges.png.

POST /api/align accepts multipart/form-data with a photo file and optional
template_id / corner_mode fields. Returns JSON with alignment result and
optionally saves artifacts to storage.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from app.api.deps import get_config, get_normalizer_service
from app.core.errors import DomainError
from app.core.schemas import TemplateSpec
from app.core.storage import LocalStorage
from app.normalizer.service import NormalizerService

router = APIRouter()


def _load_template_spec(template_arg: str, config) -> TemplateSpec | None:
    """Load a TemplateSpec from a file path or template_id directory.

    If template_arg is a path to spec.json, load it directly. Otherwise
    treat it as a template_id and look for <templates_dir>/<id>/spec.json.
    Returns None if not found (caller falls back to config defaults).
    """
    if not template_arg:
        return None
    p = Path(template_arg)
    if p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
        return TemplateSpec.model_validate(data)
    spec_path = config.templates_dir / template_arg / "spec.json"
    if spec_path.is_file():
        data = json.loads(spec_path.read_text(encoding="utf-8"))
        return TemplateSpec.model_validate(data)
    return None


@router.post("")
def align(
    photo: UploadFile = File(...),
    template_id: str = Form(None),
    corner_mode: str = Form(None),
):
    """Align a single uploaded photo.

    Steps:
      1. Reject uploads larger than max_upload_mb before reading into memory.
      2. Read raw bytes from UploadFile.
      3. Resolve TemplateSpec (from template_id or config defaults).
      4. Call NormalizerService.align() -> (aligned image, edges, result).
      5. Optionally save aligned_blank.png / edges.png via storage.
      6. Return JSON response with marker IDs, output size, corner mode.
    """
    config = get_config()
    service = get_normalizer_service(config)

    # Size check before reading into memory (protects against large uploads)
    if photo.size and photo.size > config.max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail={"code": "UploadTooLargeError", "message": "File too large"},
        )

    raw = photo.file.read()
    template_spec = _load_template_spec(template_id, config)

    try:
        output_image, edges_image, result = service.align(
            raw,
            template_spec=template_spec,
            corner_mode=corner_mode,
        )
    except DomainError as e:
        raise HTTPException(
            status_code=e.http_status,
            detail={"code": e.code, "message": e.message, "details": e.details},
        )

    # Encode artifacts once; also return them as data URLs so the Web UI can
    # preview the aligned sheet and Canny map without an extra request.
    aligned_bytes = _encode_png(output_image)
    edges_bytes = _encode_png(edges_image) if edges_image is not None else None

    sc = config.normalizer
    aligned_name = sc.output_filenames.get("aligned", "aligned_blank.png")
    edges_name = sc.output_filenames.get("edges", "edges.png")
    if sc.save_aligned:
        storage = LocalStorage(config.data_dir)
        storage.save_bytes(f"jobs/align/{aligned_name}", aligned_bytes)
    if sc.save_edges and edges_bytes is not None:
        storage = LocalStorage(config.data_dir)
        storage.save_bytes(f"jobs/align/{edges_name}", edges_bytes)

    return JSONResponse({
        "status": "ok",
        "output": aligned_name if sc.save_aligned else None,
        "edges": edges_name if sc.save_edges else None,
        "aligned_image": _to_data_url(aligned_bytes),
        "edges_image": _to_data_url(edges_bytes) if edges_bytes is not None else None,
        "marker_ids": result.marker_ids,
        "output_size": list(result.output_size),
        "corner_mode": result.corner_mode,
        "warnings": result.warnings,
    })


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