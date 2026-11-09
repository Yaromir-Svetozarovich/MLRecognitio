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

# Фиксированная высота шапки бланка в мм (заголовок + автор + инструкция + отступы).
HEADER_HEIGHT_MM = 45.0
# Минимально допустимая высота строки таблицы в мм.
MIN_ROW_HEIGHT_MM = 6.0
# Отступ перед таблицей после шапки в мм.
TABLE_GAP_MM = 5.0


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


def compute_available_height_mm(layout: Dict[str, Any]) -> float:
    """Вычислить доступную высоту для таблицы вопросов в мм.

    Берёт высоту страницы и вычитает верхний/нижний отступы, высоту шапки
    и отступ перед таблицей. Остальное место отдаётся под строки таблицы.
    """
    page = layout["page"]
    margins = layout.get("margins", {})
    page_height_mm = float(page["height_mm"])
    top_margin_mm = float(margins.get("top_mm", 32))
    bottom_margin_mm = float(margins.get("bottom_mm", 32))
    available = (
        page_height_mm
        - top_margin_mm
        - bottom_margin_mm
        - HEADER_HEIGHT_MM
        - TABLE_GAP_MM
    )
    return available


def compute_row_height_mm(layout: Dict[str, Any], num_questions: int) -> float:
    """Вычислить высоту строки таблицы в мм так, чтобы влезло N вопросов.

    Использует всё доступное пространство равномерно. Если получится меньше
    MIN_ROW_HEIGHT_MM — возвращает MIN_ROW_HEIGHT_MM (вызывающая сторона
    должна sama проверить и выдать ошибку).
    """
    available = compute_available_height_mm(layout)
    row_height = available / num_questions
    return max(row_height, MIN_ROW_HEIGHT_MM)


def check_table_width_mm(layout: Dict[str, Any]) -> None:
    """Проверить, что таблица с заданными колонками влезет по ширине листа."""
    page = layout["page"]
    margins = layout.get("margins", {})
    columns = layout.get("question_columns_mm") or [12, 112, 56]
    page_width_mm = float(page["width_mm"])
    left_margin_mm = float(margins.get("left_mm", 15))
    right_margin_mm = float(margins.get("right_mm", 15))
    total_width_mm = left_margin_mm + sum(columns) + right_margin_mm
    if total_width_mm > page_width_mm:
        raise ValidationError(
            "Table width exceeds page width",
            {
                "table_width_mm": total_width_mm,
                "page_width_mm": page_width_mm,
                "message": f"Table ({total_width_mm} mm) is wider than page ({page_width_mm} mm). "
                           f"Reduce question_columns_mm or margins.",
            },
        )


def build_roi_fields(
    layout: Dict[str, Any],
    content: Dict[str, Any],
    row_height_mm: float | None = None,
    question_order: List[int] | None = None,
) -> tuple[List[RoiField], List[int]]:
    """Build answer-field ROIs from the same geometry the renderer uses.

    The questions table is positioned at ``layout.roi.table_top_mm`` (floating
    table, page-relative). Each row has a fixed height ``row_height_mm`` and
    the answer cell starts after the first two columns. Therefore ROI
    coordinates can be derived deterministically without measuring the DOCX.

    Editing ``template_layout.json`` (``roi.table_top_mm`` /
    ``roi.row_height_mm`` / ``question_columns_mm``) changes both the rendered
    table and the resulting crops, as required by ЛР3.

    Args:
        layout: parsed template_layout.json.
        content: parsed content dict with questions list.
        row_height_mm: fixed row height in mm. If None, computed automatically.
        question_order: original question numbers in the order they appear
            on the sheet. If None, uses sequential 1..N.

    Returns:
        Tuple of (list of RoiField, question_order list).
    """
    roi_cfg = layout.get("roi") or {}
    table_top_mm = float(roi_cfg.get("table_top_mm", 78))
    if row_height_mm is None:
        # Use fixed value from layout if present, otherwise auto-compute
        row_height_mm = float(roi_cfg.get("row_height_mm") or 0)
        if row_height_mm <= 0:
            row_height_mm = compute_row_height_mm(layout, len(content["questions"]))
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

    if question_order is None:
        question_order = [q["num"] for q in content["questions"]]

    return fields, question_order


def parse_csv_questions(
    csv_text: str,
    min_questions: int = 15,
    max_questions: int = 20,
    num_questions: int | None = None,
) -> List[Dict[str, Any]]:
    """Parse a CSV with questions and return a validated list of question dicts.

    Expected CSV columns: num, text, answer_type.
    answer_type must be one of: letter, digit, number, word, sentence.

    The parser is tolerant of commas inside question text: if a field is not
    quoted, it assumes the last comma separates the answer_type and everything
    between the first comma (after num) and the last comma is the question text.

    Args:
        csv_text: raw CSV text (UTF-8).
        min_questions: minimum number of questions in the variant.
        max_questions: maximum number of questions in the variant.
        num_questions: if set, exactly this many questions are selected
            (must be <= available questions in bank).

    Returns:
        List of question dicts with keys: num, text, answer_type.

    Raises:
        ValidationError: if CSV is malformed, missing columns, or cannot build
            a balanced variant, or num_questions exceeds available questions.
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

    # If num_questions is specified, validate against available count
    if num_questions is not None:
        if num_questions < 1:
            raise ValidationError("num_questions must be >= 1")
        if num_questions > len(rows):
            raise ValidationError(
                f"Not enough questions in bank: requested {num_questions}, "
                f"available {len(rows)}",
                {
                    "requested": num_questions,
                    "available": len(rows),
                },
            )

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

    # 1) Take one random question of each type (guarantee all types present)
    for t in ALLOWED_ANSWER_TYPES:
        pool = by_type[t]
        q = rng.choice(pool)
        chosen.append(q)
        used_indices.add(id(q))

    # 2) Fill remaining up to target count
    if num_questions is not None:
        target = num_questions
    else:
        target = rng.randint(min_questions, max_questions)

    remaining_pool = [q for q in rows if id(q) not in used_indices]
    rng.shuffle(remaining_pool)
    needed = target - len(chosen)
    if needed > 0 and remaining_pool:
        chosen.extend(remaining_pool[:needed])

    # 3) Shuffle for variant randomness, then sort by number and renumber
    rng.shuffle(chosen)
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
    num_questions: int | None = None,
) -> Dict[str, Any]:
    """Build a content dict from CSV text for template generation.

    Args:
        csv_text: raw CSV text with questions.
        header: optional header override (author, group, variant, course).
        instruction: optional instruction override.
        version: template version number.
        version_marker_id: ArUco marker ID encoding the version (BL corner).
        num_questions: if set, exactly this many questions are selected
            from the bank (validated against available count).

    Returns:
        Content dict compatible with validate_inputs / build_template_spec.
    """
    questions = parse_csv_questions(csv_text, num_questions=num_questions)
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
      - every question has num, text, answer_type
      - all 5 answer types are present: letter, digit, number, word, sentence
      - table width fits on page

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

    # Check table width fits on page
    try:
        check_table_width_mm(layout)
    except ValidationError as e:
        errs.append(e.message)

    for k in ("version_marker_id", "header", "instruction", "questions"):
        if k not in content:
            errs.append(f"content missing key: {k}")
    if errs:
        raise ValidationError("Validation failed", {"errors": errs})

    vm = content["version_marker_id"]
    if not isinstance(vm, int) or vm in (0, 1, 2) or vm > 49:
        errs.append("version_marker_id must be int in 3..49 (0,1,2 reserved for corners)")

    qs = content["questions"]
    if len(qs) < 1:
        errs.append("questions count must be >= 1, got 0")

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
      - row_height_mm: computed row height in mm
      - question_order: original question numbers in sheet order
      - question_count: number of questions on the sheet

    The BL marker ID is set to content.version_marker_id, which encodes
    the template version in the physical sheet.

    Args:
        layout: parsed template_layout.json.
        content: parsed content dict with questions list.
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

    # Compute row height automatically so all questions fit on one A4 page,
    # unless layout.roi.row_height_mm is explicitly set.
    num_questions = len(content["questions"])
    fixed_row_height = float(roi_cfg.get("row_height_mm") or 0)
    if fixed_row_height > 0:
        row_height_mm = fixed_row_height
    else:
        row_height_mm = compute_row_height_mm(layout, num_questions)
        # Check that row height is not too small (only for auto-computed)
        if row_height_mm <= MIN_ROW_HEIGHT_MM + 0.5:
            raise ValidationError(
                "Too many questions for one A4 page",
                {
                    "num_questions": num_questions,
                    "row_height_mm": row_height_mm,
                    "message": f"Not enough vertical space for {num_questions} questions. "
                               f"Row height would be {row_height_mm:.1f} mm "
                               f"(minimum {MIN_ROW_HEIGHT_MM} mm). "
                               f"Reduce number of questions.",
                },
            )

    # Build ROI fields with computed row height
    roi_fields, question_order = build_roi_fields(
        layout, content,
        row_height_mm=row_height_mm,
    )

    return TemplateSpec(
        template_version=template_id,
        dictionary_id=dictionary_id,
        expected_marker_ids=expected_marker_ids,
        corner_by_id=corner_by_id,  # type: ignore[arg-type]
        marker_layout=marker_layout,
        marker_size_px=mm_to_px(mc["size_mm"]),
        canvas_size=canvas_size,
        roi_fields=roi_fields,
        padding_px=int(roi_cfg.get("padding_px", 5)),
        question_order=question_order,
        question_count=num_questions,
        row_height_mm=row_height_mm,
    )
