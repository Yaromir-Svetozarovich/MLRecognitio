"""ROI specification loader.

Reads a TemplateSpec (from file or object) and exposes its roi_fields with
validation. This is the single entry point for obtaining ROI coordinates
from a template — normalizer and extractor never parse raw JSON themselves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from app.core.config import AppConfig
from app.core.errors import CropOutOfBoundsError, JobNotFoundError
from app.core.schemas import RoiField, TemplateSpec


def load_template_spec(template_arg: str, config: AppConfig) -> TemplateSpec:
    """Load a TemplateSpec from a file path or template_id directory.

    Args:
        template_arg: either a path to spec.json or a template_id.
        config: AppConfig with templates_dir.

    Returns:
        Validated TemplateSpec.

    Raises:
        JobNotFoundError: if template_id is not found on disk.
        ValueError: if JSON is invalid.
    """
    p = Path(template_arg)
    if p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
        return TemplateSpec.model_validate(data)
    spec_path = config.templates_dir / template_arg / "spec.json"
    if spec_path.is_file():
        data = json.loads(spec_path.read_text(encoding="utf-8"))
        return TemplateSpec.model_validate(data)
    raise JobNotFoundError(details={"template": template_arg})


def get_roi_fields(spec: TemplateSpec) -> list[RoiField]:
    """Return the list of RoiField from a TemplateSpec.

    Returns an empty list if roi_fields is not populated (ЛР1 templates
    may not have ROIs yet).
    """
    return list(spec.roi_fields)


def validate_roi_fields(spec: TemplateSpec) -> list[str]:
    """Validate that all ROI fields fit within the canvas.

    Returns a list of error strings (empty if all valid). Does not raise;
    callers decide whether to raise CropOutOfBoundsError.
    """
    errors: list[str] = []
    canvas_w, canvas_h = spec.canvas_size
    for field in spec.roi_fields:
        x, y, w, h = field.coords
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            errors.append(f"ROI '{field.id}': invalid coords {field.coords}")
            continue
        if x + w > canvas_w or y + h > canvas_h:
            errors.append(
                f"ROI '{field.id}': coords {field.coords} exceed canvas {spec.canvas_size}"
            )
    return errors