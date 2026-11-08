"""Template spec generation (source of truth about a correct sheet).

Reads layout.json + content.json (or CSV with questions), validates them,
and produces a TemplateSpec that is the single source of truth for the
normalizer and recognizer domains.
"""

from __future__ import annotations

import csv
import io
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List

import cv2

from app.core.errors import ValidationError
from app.core.geometry import mm_to_px
from app.core.schemas import RoiField, TemplateSpec

DICT_NAME_TO_ID = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
}

ALLOWED_ANSWER_TYPES = {"letter", "digit", "number", "word", "sentence"}

# Canonical marker -> corner mapping (SKILL v4 §3 / README):
#   ID 0 -> TL, ID 1 -> TR, ID 2 -> BR, version_marker_id -> BL.
# The BL marker carries the template version, the other three are fixed.
MARKER_CORNERS = ("TL", "TR", "BR", "BL")

# Answer-type -> ROI field type expected by the recogniser.
ROI_TYPE_BY_ANSWER = {
    "digit": "digit",
    "number": "digit",
    "letter": "text",
    "word": "text",
    "sentence": "handwriting",
}


def resolve_marker_ids(layout: Dict[str, Any], content: Dict[str, Any]) -> Dict[str, int]:
    """Return the physical marker ID for each page corner.

    This is the single place that decides the ArUco layout, shared by the
    spec builder and the DOCX renderer so they can never drift apart.

    Mapping (SKILL v4): TL, TR and BR come from ``layout.markers.ids``;
    BL is always ``content.version_marker_id`` (the version marker).
    """
    mc = layout["markers"]
    return {
        "TL": int(mc["ids"]["TL"]),
        "TR": int(mc["ids"]["TR"]),
        "BR": int(mc["ids"]["BR"]),
        "BL": int(content["version_marker_id"]),
    }


def build_roi_fields(layout: Dict[str, Any], content: Dict[str, Any]) -> List[RoiField]:
    """Build answer-field ROIs from the same geometry the renderer uses.

    The questions table is positioned at ``layout.roi.table_top_mm`` (floating
    table, page-relative). Each row has a fixed height ``row_height_mm`` and
    the answer cell starts after the first two columns. Therefore ROI
    coordinates can be derived deterministically without measuring the DOCX.

    Editing ``template_layout.json`` (``roi.table_top_mm`` /
    ``roi.row_height_mm`` / ``question_columns_mm``) changes both the rendered
    table and the resulting crops, as required by ЛР3.
    """
    roi_cfg = layout.get("roi") or {}
    table_top_mm = float(roi_cfg.get("table_top_mm", 78))
    row_height_mm = float(roi_cfg.get("row_height_mm", 7))
    columns = layout.get("question_columns_mm") or [12, 112, 56]
    left_mm = float(layout.get("margins", {}).get("left_mm", 15))

    answer_x_mm = left_mm + float(columns[0]) + float(columns[1])
    answer_w_mm = float(columns[2])

    fields: List[RoiField] = []
    for idx, q in enumerate(content["questions"]):
        y_mm = table_top_mm + idx * row_height_mm
        coords = (
            mm_to_px(answer_x_mm),
            mm_to_px(y_mm),
            mm_to_px(answer_w_mm),
            mm_to_px(row_height_mm),
        )
        fields.append(
            RoiField(
                id=f"q{q['num']}_answer",
                type=ROI_TYPE_BY_ANSWER.get(q["answer_type"], "text"),
                coords=coords,
            )
        )
    return fields


def parse_csv_questions(csv_text: str, min_questions: int = 15, max_questions: int = 20) -> List[Dict[str, Any]]:
    """Parse a CSV with questions and return a validated list of question dicts.

    Expected CSV columns: num, text, answer_type.
    answer_type must be one of: letter, digit, number, word, sentence.

    Randomly selects min_questions..max_questions questions so that all 5
    answer types are represented at least once (balanced variant).

    The parser is tolerant of commas inside question text: if a field is not
    quoted, it assumes the last comma separates the answer_type and everything
    between the first comma (after num) and the last comma is the question text.

    Args:
        csv_text: raw CSV text (UTF-8).
        min_questions: minimum number of questions in the variant.
        max_questions: maximum number of questions in the variant.

    Returns:
        List of question dicts with keys: num, text, answer_type.

    Raises:
        ValidationError: if CSV is malformed, missing columns, or cannot build
            a balanced variant.
    """
    lines = [ln for ln in csv_text.splitlines() if ln.strip()]
    if not lines:
        raise ValidationError("CSV is empty", {"expected": ["num,text,answer_type"]})

    header = [h.strip() for h in lines[0].split(",")]
    required = {"num", "text", "answer_type"}
    missing = required - set(header)
    if missing:
        raise ValidationError("CSV missing columns", {"missing": sorted(missing)})

    num_idx = header.index("num")
    text_idx = header.index("text")
    type_idx = header.index("answer_type")

    rows: List[Dict[str, Any]] = []
    for i, line in enumerate(lines[1:], start=2):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue  # skip malformed rows
        num = parts[0]
        at = parts[-1].strip().strip('"').lower()
        # Text is everything between first and last column
        text = ", ".join(parts[1:-1]).strip().strip('"')

        if not num or not text or not at:
            continue
        if at not in ALLOWED_ANSWER_TYPES:
            raise ValidationError(
                f"Invalid answer_type on row {i}: {at}",
                {"allowed": sorted(ALLOWED_ANSWER_TYPES)},
            )
        try:
            num_int = int(num)
        except ValueError:
            raise ValidationError(f"Invalid num on row {i}: {num}")
        rows.append({"num": num_int, "text": text, "answer_type": at})

    if not rows:
        raise ValidationError("CSV contains no valid questions")

    # Balanced variant: guarantee all 5 types, then fill randomly
    by_type: Dict[str, List[Dict[str, Any]]] = {t: [] for t in ALLOWED_ANSWER_TYPES}
    for q in rows:
        by_type[q["answer_type"]].append(q)

    missing_types = [t for t in ALLOWED_ANSWER_TYPES if not by_type[t]]
    if missing_types:
        raise ValidationError(
            f"CSV missing answer types: {missing_types}",
            {"available": [t for t in ALLOWED_ANSWER_TYPES if by_type[t]]},
        )

    rng = random.Random()
    chosen: List[Dict[str, Any]] = []
    used_indices: set[int] = set()

    # 1) Take one random question of each type
    for t in ALLOWED_ANSWER_TYPES:
        pool = by_type[t]
        q = rng.choice(pool)
        chosen.append(q)
        used_indices.add(id(q))

    # 2) Fill remaining up to random count in [min_questions, max_questions]
    target = rng.randint(min_questions, max_questions)
    remaining_pool = [q for q in rows if id(q) not in used_indices]
    rng.shuffle(remaining_pool)
    needed = target - len(chosen)
    if needed > 0 and remaining_pool:
        chosen.extend(remaining_pool[:needed])

    # 3) Sort by number and renumber sequentially
    chosen.sort(key=lambda q: q["num"])
    for idx, q in enumerate(chosen, 1):
        q["num"] = idx

    return chosen


def content_from_csv(
    csv_text: str,
    header: Dict[str, Any] | None = None,
    instruction: str | None = None,
    version: int = 1,
    version_marker_id: int = 3,
) -> Dict[str, Any]:
    """Build a content dict from CSV text for template generation.

    Args:
        csv_text: raw CSV text with questions.
        header: optional header override (author, group, variant, course).
        instruction: optional instruction override.
        version: template version number.
        version_marker_id: ArUco marker ID encoding the version (BL corner).

    Returns:
        Content dict compatible with validate_inputs / build_template_spec.
    """
    questions = parse_csv_questions(csv_text)
    return {
        "version": version,
        "version_marker_id": version_marker_id,
        "header": header or {
            "author": "Автор",
            "group": "Группа",
            "variant": str(version),
            "course": "1",
        },
        "instruction": instruction or (
            "Заполняйте печатными заглавными буквами, используйте только "
            "синюю или чёрную ручку; ответ пишите в поле справа от вопроса."
        ),
        "questions": questions,
    }


def validate_inputs(layout: Dict[str, Any], content: Dict[str, Any]) -> None:
    """Validate layout.json and content.json before building TemplateSpec.

    Collects all errors and raises a single ValidationError with the full
    list, so the user sees everything at once.

    Checks performed:
      - layout.markers.positions contains TL, TR, BL, BR
      - marker size_mm and quiet_zone_mm are > 0
      - content has version_marker_id, header, instruction, questions
      - version_marker_id is int in 3..49 (0,1,2 reserved for corner markers)
      - questions count is 15..20
      - every question has num, text, answer_type
      - all 5 answer types are present: letter, digit, number, word, sentence

    Raises:
        ValidationError: with details.errors list if any check fails.
    """
    errs: List[str] = []

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
    need = ALLOWED_ANSWER_TYPES
    miss = need - types_seen
    if miss:
        errs.append(f"missing answer types: {sorted(miss)}")

    if errs:
        raise ValidationError("Validation failed", {"errors": errs})


def build_template_spec(
    layout: Dict[str, Any],
    content: Dict[str, Any],
    template_id: str | None = None,
) -> TemplateSpec:
    """Build a TemplateSpec from validated layout + content.

    Computes:
      - dictionary_id from layout.markers.dict
      - marker_layout: marker ID -> (x, y) centre pixel on 2100x2970 canvas
      - corner_by_id: marker ID -> corner name (TL/TR/BR/BL)
      - expected_marker_ids: the 4 IDs in order [TL, TR, BR, BL]
      - marker_size_px: physical marker size in pixels
      - canvas_size: page size in pixels
      - roi_fields: answer-cell ROIs derived from the same layout geometry

    The BL marker ID is set to content.version_marker_id, which encodes
    the template version in the physical sheet.

    Args:
        layout: parsed template_layout.json.
        content: parsed content_vN.json.
        template_id: optional explicit template version string.

    Returns:
        TemplateSpec populated with all computed fields.

    Raises:
        ValidationError: if inputs are invalid or dictionary unknown.
    """
    validate_inputs(layout, content)

    mc = layout["markers"]
    dict_name = mc.get("dict", "DICT_4X4_50")
    if dict_name not in DICT_NAME_TO_ID:
        raise ValidationError("Unknown dictionary", {"dict": dict_name})
    dictionary_id = DICT_NAME_TO_ID[dict_name]

    total_mm = mc["size_mm"] + 2 * mc["quiet_zone_mm"]
    half_total_mm = total_mm / 2.0

    corner_by_id: Dict[int, str] = {}
    marker_layout: Dict[int, tuple[int, int]] = {}

    marker_ids = resolve_marker_ids(layout, content)
    for corner, mid in marker_ids.items():
        pos = mc["positions"][corner]
        cx_mm = pos["x_mm"] + half_total_mm
        cy_mm = pos["y_mm"] + half_total_mm
        marker_layout[mid] = (mm_to_px(cx_mm), mm_to_px(cy_mm))
        corner_by_id[mid] = corner

    expected_marker_ids = [marker_ids[corner] for corner in MARKER_CORNERS]

    if template_id is None:
        template_id = f"template_v{content.get('version', 0)}_{uuid.uuid4().hex[:8]}"

    page = layout["page"]
    canvas_size = (mm_to_px(page["width_mm"]), mm_to_px(page["height_mm"]))
    roi_cfg = layout.get("roi") or {}

    return TemplateSpec(
        template_version=template_id,
        dictionary_id=dictionary_id,
        expected_marker_ids=expected_marker_ids,
        corner_by_id=corner_by_id,  # type: ignore[arg-type]
        marker_layout=marker_layout,
        marker_size_px=mm_to_px(mc["size_mm"]),
        canvas_size=canvas_size,
        roi_fields=build_roi_fields(layout, content),
        padding_px=int(roi_cfg.get("padding_px", 5)),
    )