"""Normalizer CV-node: alignment via ArUco markers + perspective warp (этап 3).

This is aligner.py — a pure CV node, not the DAG pipeline.
No I/O, no console output; returns numpy arrays + Pydantic AlignResult.
"""

from __future__ import annotations

import cv2
import numpy as np
from pydantic import TypeAdapter

from app.core.errors import ConfigError, HomographyError, WarpError
from app.core.schemas import AlignConfig, AlignResult, TemplateSpec
from app.normalizer import markers, preprocessing


def _quad_area(points: np.ndarray) -> float:
    """Compute the area of a quadrilateral using the shoelace formula.

    Used to detect degenerate (zero-area or near-zero-area) marker
    configurations before attempting getPerspectiveTransform.
    """
    pts = np.asarray(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2.0


def align(
    image: np.ndarray,
    config: AlignConfig,
    template_spec: TemplateSpec | None = None,
) -> tuple[np.ndarray, np.ndarray | None, AlignResult]:
    """Run full alignment: preprocess -> detect markers -> validate -> homography -> warp.

    Args:
        image: BGR numpy array (the photo of the filled blank).
        config: AlignConfig with dictionary, output size, Canny thresholds, corner_mode.
        template_spec: Optional TemplateSpec — authoritative source for
            expected_marker_ids and corner_by_id. If None, falls back to
            config.expected_ids (CLI debug mode).

    Returns:
        (output_image, edges_image, align_result) where:
          - output_image: warped BGR image, exactly config.output_size
          - edges_image: Canny edge map (may be None)
          - align_result: Pydantic AlignResult with metadata

    Raises:
        HomographyError: if points are degenerate or H is invalid.
        WarpError: if warpPerspective fails or output size is wrong.
        ConfigError: if centers_to_layout requested without TemplateSpec.
    """
    if image is None:
        raise ValueError("image is None")

    # Этап 1: предобработка
    gray, edges = preprocessing.preprocess(image, config.canny_thresholds)

    # Этап 2: локализация маркеров
    corners, ids = markers.detect_markers(image, config.dictionary_id)

    # Determine expected IDs and corner mapping from TemplateSpec or config fallback
    if template_spec is not None:
        expected_ids = template_spec.expected_marker_ids
        corner_by_id = template_spec.corner_by_id
    else:
        expected_ids = list(config.expected_ids)
        corner_by_id = {
            expected_ids[0]: "TL",
            expected_ids[1]: "TR",
            expected_ids[2]: "BR",
            expected_ids[3]: "BL",
        }

    flat_ids = markers.validate_markers(ids, expected_ids)

    corner_mode = config.corner_mode
    if corner_mode == "centers_to_layout" and template_spec is None:
        raise ConfigError(
            "corner_mode 'centers_to_layout' requires a TemplateSpec",
            details={"corner_mode": corner_mode},
        )

    # Build source points (marker locations in the photo)
    src, ordered_ids = markers.build_src_points(corners, ids, corner_by_id, corner_mode)

    # Build destination points (where they map on the aligned canvas)
    w, h = config.output_size
    if corner_mode == "centers_to_layout":
        if template_spec is None or template_spec.marker_layout is None:
            raise ConfigError("TemplateSpec with marker_layout required for centers_to_layout")
        layout = template_spec.marker_layout
        dst = np.array([layout[mid] for mid in ordered_ids], dtype=np.float32)
    else:
        # centers or outer_corners: map to canvas corners
        dst = np.array(
            [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)],
            dtype=np.float32,
        )

    # Degeneracy checks before homography
    if not np.all(np.isfinite(src)) or not np.all(np.isfinite(dst)):
        raise HomographyError(details={"reason": "non-finite coordinates"})
    area = _quad_area(src)
    if area < config.min_quad_area_px:
        raise HomographyError(details={"area": area, "min": config.min_quad_area_px})

    # Compute perspective transform
    try:
        H = cv2.getPerspectiveTransform(src, dst)
    except cv2.error as e:
        raise HomographyError(details={"opencv": str(e)})

    if not np.all(np.isfinite(H)):
        raise HomographyError(details={"reason": "non-finite homography"})
    if np.linalg.matrix_rank(H) < 3:
        raise HomographyError(details={"reason": "rank < 3"})

    # Apply warp
    try:
        warped = cv2.warpPerspective(
            image, H, (w, h), borderValue=config.border_value
        )
    except cv2.error as e:
        raise WarpError(details={"opencv": str(e)})

    if warped is None or warped.shape[1] != w or warped.shape[0] != h:
        raise WarpError(details={"shape": None if warped is None else warped.shape})

    result = AlignResult(
        marker_ids=ordered_ids,
        source_points=TypeAdapter(list[tuple[float, float]]).validate_python(
            src.tolist()
        ),
        dst_points=TypeAdapter(list[tuple[float, float]]).validate_python(
            dst.tolist()
        ),
        homography=TypeAdapter(list[list[float]]).validate_python(H.tolist()),
        output_size=(w, h),
        corner_mode=corner_mode,
        warnings=[],
    )
    return warped, edges, result