"""CLI for template generation (thin adapter).

Entry point: python generate_blank.py (delegates here) or
python -m app.template.cli --layout ... --content ... --output ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.template.generator import build_template_spec, validate_inputs
from app.template.renderer import generate_blank


def main(argv=None) -> int:
    """Parse args, validate layout+content, build TemplateSpec, render DOCX,
    and save spec.json next to the output DOCX."""
    ap = argparse.ArgumentParser(description="Generate ArUco-annotated answer blank (DOCX)")
    ap.add_argument("--layout", required=True)
    ap.add_argument("--content", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--template-id", default=None)
    args = ap.parse_args(argv)

    layout_path = Path(args.layout)
    content_path = Path(args.content)
    output_path = Path(args.output)

    with layout_path.open(encoding="utf-8") as f:
        layout = json.load(f)
    with content_path.open(encoding="utf-8") as f:
        content = json.load(f)

    try:
        validate_inputs(layout, content)
        spec = build_template_spec(layout, content, template_id=args.template_id)
    except Exception as e:
        print(f"Validation failed: {e}")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_blank(layout, content, str(output_path))

    spec_path = output_path.with_suffix(".spec.json")
    spec_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")

    from docx import Document
    Document(str(output_path))  # sanity: reopen the produced DOCX
    print(f"OK: {output_path}")
    print(f"SPEC: {spec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())