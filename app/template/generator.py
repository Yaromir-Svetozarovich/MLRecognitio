"""Template spec generation (source of truth about a correct sheet)."""

from __future__ import annotations

import cv2
import uuid
from datetime import datetime, timezone
from typing import Any

from app.core.errors import ValidationError
from app.core.geometry import mm_to_px
from app.core.schemas import TemplateSpec


DICT_NAME_TO_ID = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
}


def validate_inputs(layout: dict[str, Any], content: dict[str, Any]) -> None:
    errs: list[str] = []

    try:
        m = layout["markers"]
        if set(m["positions"].keys()) != {"TL", "TR", "BL", "BR"}:
            errs.append("markers.positions must contain TL, TR, BL, BR")
        if m["size_mm"] <= 0 or m["quiet_zone_mm"] <= 0:
            errs.append("marker size and quiet_zone must be > 0")
    except KeyError as e:
        errs.append(f"layout missing key: {e}")

    for k in ("version_marker_id", "header", "instruction", "questions"):
        if k not in content:
            errs.append(f"content missing key: {k}")
    if errs:
        raise ValidationError("Validation failed", {"errors": errs})

    vm = content["version_marker_id"]
    if not isinstance(vm, int) or vm in (0, 1, 2) or vm > 49:
        errs.append("version_marker_id must be int in 3..49 (0,1,2 reserved for corners)")

    qs = content["questions"]
    if not (15 <= len(qs) <= 20):
        errs.append(f"questions count must be 15..20, got {len(qs)}")

    types_seen: set[str] = set()
    for i, q in enumerate(qs):
        for k in ("num", "text", "answer_type"):
            if k not in q:
                errs.append(f"question[{i}] missing {k}")
        types_seen.add(q.get("answer_type"))
    need = {"letter", "digit", "number", "word", "sentence"}
    miss = need - types_seen
    if miss:
        errs.append(f"missing answer types: {sorted(miss)}")

    if errs:
        raise ValidationError("Validation failed", {"errors": errs})


def build_template_spec(
    layout: dict[str, Any],
    content: dict[str, Any],
    template_id: str | None = None,
) -> TemplateSpec:
    validate_inputs(layout, content)

    mc = layout["markers"]
    dict_name = mc.get("dict", "DICT_4X4_50")
    if dict_name not in DICT_NAME_TO_ID:
        raise ValidationError("Unknown dictionary", {"dict": dict_name})
    dictionary_id = DICT_NAME_TO_ID[dict_name]

    total_mm = mc["size_mm"] + 2 * mc["quiet_zone_mm"]
    half_total_mm = total_mm / 2.0

    corner_by_id: dict[int, str] = {}
    marker_layout: dict[int, tuple[int, int]] = {}

    marker_ids = {
        "TL": mc["ids"]["TL"],
        "TR": mc["ids"]["TR"],
        "BL": mc["ids"]["BL"],
        "BR": content["version_marker_id"],
    }
    for corner, mid in marker_ids.items():
        pos = mc["positions"][corner]
        cx_mm = pos["x_mm"] + half_total_mm
        cy_mm = pos["y_mm"] + half_total_mm
        marker_layout[mid] = (mm_to_px(cx_mm), mm_to_px(cy_mm))
        corner_by_id[mid] = corner

    expected_marker_ids = [
        mc["ids"]["TL"],
        mc["ids"]["TR"],
        mc["ids"]["BL"],
        content["version_marker_id"],
    ]

    if template_id is None:
        template_id = f"template_v{content.get('version', 0)}_{uuid.uuid4().hex[:8]}"

    page = layout["page"]
    canvas_size = (mm_to_px(page["width_mm"]), mm_to_px(page["height_mm"]))

    return TemplateSpec(
        template_id=template_id,
        version=str(content.get("version", "1")),
        dictionary_id=dictionary_id,
        expected_marker_ids=expected_marker_ids,
        corner_by_id=corner_by_id,  # type: ignore[arg-type]
        marker_layout=marker_layout,
        marker_size_px=mm_to_px(mc["size_mm"]),
        canvas_size=canvas_size,
        rois=[],
        created_at=datetime.now(timezone.utc),
    )