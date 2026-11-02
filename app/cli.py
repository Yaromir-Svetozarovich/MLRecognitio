"""Thin CLI adapter for the template generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.errors import DomainError
from app.template.generator import build_template_spec, validate_inputs
from app.template.renderer import generate_blank


def main(argv=None) -> int:
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
    except DomainError as e:
        print(f"Validation failed: {e.message}")
        if e.details:
            for err in e.details.get("errors", []):
                print(f"  - {err}")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_blank(layout, content, str(output_path))

    # Save TemplateSpec next to output for future integration.
    spec_path = output_path.with_suffix(".spec.json")
    spec_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")

    # Sanity reopen
    from docx import Document
    Document(str(output_path))
    print(f"OK: {output_path}")
    print(f"SPEC: {spec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())