"""Tests for the template generator (ЛР1).

Covers input validation and TemplateSpec construction from layout + content.
"""

import json
from pathlib import Path

import pytest

from app.core.errors import DomainError
from app.template.generator import build_template_spec, validate_inputs


def load_fixtures():
    """Load template_layout.json and content_v1.json from the project root."""
    root = Path(__file__).resolve().parents[1]
    layout = json.loads((root / "template_layout.json").read_text(encoding="utf-8"))
    content = json.loads((root / "content_v1.json").read_text(encoding="utf-8"))
    return layout, content


def test_validate_inputs_ok():
    """Valid layout + content should pass validation without raising."""
    layout, content = load_fixtures()
    validate_inputs(layout, content)  # no exception


def test_validate_inputs_bad_count():
    """Fewer than 15 questions should raise ValidationError."""
    layout, content = load_fixtures()
    content["questions"] = content["questions"][:5]
    with pytest.raises(DomainError):
        validate_inputs(layout, content)


def test_build_template_spec():
    """TemplateSpec must have correct canvas size, marker size, layout, corners."""
    layout, content = load_fixtures()
    spec = build_template_spec(layout, content, template_id="t1")
    assert spec.template_version == "t1"
    assert spec.canvas_size == (2100, 2970)
    assert spec.marker_size_px == 150
    assert set(spec.corner_by_id.values()) == {"TL", "TR", "BL", "BR"}
    assert len(spec.expected_marker_ids) == 4
    assert 0 in spec.marker_layout and content["version_marker_id"] in spec.marker_layout
    # center of TL marker: 5mm + 12.5mm = 17.5mm -> 175px
    assert spec.marker_layout[0] == (175, 175)


def test_corner_mapping_matches_canonical_layout():
    """Marker IDs must follow SKILL v4: 0=TL, 1=TR, 2=BR, version=BL."""
    layout, content = load_fixtures()
    spec = build_template_spec(layout, content, template_id="t1")
    version_id = content["version_marker_id"]

    assert spec.corner_by_id[0] == "TL"
    assert spec.corner_by_id[1] == "TR"
    assert spec.corner_by_id[2] == "BR"
    assert spec.corner_by_id[version_id] == "BL"

    # BR is bottom-right, version marker is bottom-left on the canvas.
    assert spec.marker_layout[2] == (1925, 2795)
    assert spec.marker_layout[version_id] == (175, 2795)


def test_build_template_spec_populates_roi_fields():
    """ROIs must be generated for every question and stay inside the canvas."""
    layout, content = load_fixtures()
    spec = build_template_spec(layout, content, template_id="t1")

    assert len(spec.roi_fields) == len(content["questions"])
    assert spec.roi_fields[0].id == "q1_answer"
    assert spec.roi_fields[0].type == "text"  # content_v1 q1 is "letter"
    # Answer column starts at 15 + 12 + 112 = 139 mm -> 1390 px,
    # table top is 78 mm -> 780 px, row height 7 mm -> 70 px.
    assert spec.roi_fields[0].coords == (1390, 780, 560, 70)
    assert spec.roi_fields[1].coords == (1390, 850, 560, 70)
    assert spec.padding_px == 5

    canvas_w, canvas_h = spec.canvas_size
    for field in spec.roi_fields:
        x, y, w, h = field.coords
        assert x >= 0 and y >= 0 and w > 0 and h > 0
        assert x + w <= canvas_w
        assert y + h <= canvas_h