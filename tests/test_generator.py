import json
from pathlib import Path

import pytest

from app.core.errors import DomainError
from app.template.generator import build_template_spec, validate_inputs


def load_fixtures():
    root = Path(__file__).resolve().parents[1]
    layout = json.loads((root / "template_layout.json").read_text(encoding="utf-8"))
    content = json.loads((root / "content_v1.json").read_text(encoding="utf-8"))
    return layout, content


def test_validate_inputs_ok():
    layout, content = load_fixtures()
    validate_inputs(layout, content)  # no exception


def test_validate_inputs_bad_count():
    layout, content = load_fixtures()
    content["questions"] = content["questions"][:5]
    with pytest.raises(DomainError):
        validate_inputs(layout, content)


def test_build_template_spec():
    layout, content = load_fixtures()
    spec = build_template_spec(layout, content, template_id="t1")
    assert spec.template_id == "t1"
    assert spec.canvas_size == (2100, 2970)
    assert spec.marker_size_px == 150
    assert set(spec.corner_by_id.values()) == {"TL", "TR", "BL", "BR"}
    assert len(spec.expected_marker_ids) == 4
    assert 0 in spec.marker_layout and content["version_marker_id"] in spec.marker_layout
    # center of TL marker: 5mm + 12.5mm = 17.5mm -> 175px
    assert spec.marker_layout[0] == (175, 175)