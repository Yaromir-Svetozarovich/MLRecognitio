"""Shared data contracts (Pydantic v2 models) used by API, domains, and tests.

These schemas are the single source of truth for data shapes across the system.
Domains do not import FastAPI; they only use these Pydantic models and plain
dataclasses / numpy arrays.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Protocol

import cv2
import numpy as np
from pydantic import BaseModel, Field


class RoiField(BaseModel):
    """A single answer-field region of interest on the aligned 2100x2970 canvas.

    Attributes:
        id:                unique field identifier (matches AnswerKey key)
        type:              kind of content expected (digit, text, checkbox, handwriting)
        coords:            (x, y, w, h) rectangle in aligned-image pixel coordinates
        confidence_threshold: optional per-field override of the global confidence threshold
    """
    id: str
    type: Literal["digit", "text", "checkbox", "handwriting"]
    coords: tuple[int, int, int, int]  # x, y, w, h in 2100x2970
    confidence_threshold: float | None = None


class TemplateSpec(BaseModel):
    """Single source of truth describing a "correct" A4 blank sheet.

    Created by the generator (ЛР1), read by the normalizer (ЛР2) for marker
    validation, and read by the recognizer (ЛР3+) for ROI coordinates.

    Attributes:
        template_version:   unique template identifier (e.g. "template_v1_abc123")
        canvas_size:        output canvas size in pixels (2100, 2970)
        roi_fields:         list of answer-field ROIs (populated on ЛР3)
        dictionary_id:      ArUco dictionary used (DICT_4X4_50)
        expected_marker_ids: list of the 4 marker IDs expected on the sheet
        corner_by_id:       mapping marker ID -> corner (TL/TR/BR/BL) for orientation
        marker_layout:      marker ID -> (x, y) centre on the canvas (for centers_to_layout mode)
        marker_size_px:     physical marker size in pixels on the canvas
        padding_px:         padding around markers / fields
    """
    template_version: str
    canvas_size: tuple[int, int] = (2100, 2970)
    roi_fields: list[RoiField] = Field(default_factory=list)
    dictionary_id: int = cv2.aruco.DICT_4X4_50
    expected_marker_ids: list[int] = Field(default_factory=lambda: [0, 1, 2, 3])
    corner_by_id: dict[int, Literal["TL", "TR", "BR", "BL"]] = Field(
        default_factory=lambda: {0: "TL", 1: "TR", 2: "BR", 3: "BL"}
    )
    marker_layout: dict[int, tuple[int, int]] | None = None
    marker_size_px: int | None = None
    padding_px: int = 5
    question_order: list[int] | None = None
    question_count: int | None = None
    row_height_mm: float | None = None


class AlignConfig(BaseModel):
    """Runtime configuration for the alignment node.

    Separate from TemplateSpec: this is the *operational* config, while
    TemplateSpec describes the *sheet design*. expected_ids is a fallback
    used only when no TemplateSpec is provided (CLI debug mode).
    """
    dictionary_id: int = cv2.aruco.DICT_4X4_50
    expected_ids: tuple[int, ...] = (0, 1, 2, 3)
    output_size: tuple[int, int] = (2100, 2970)
    border_value: tuple[int, int, int] = (255, 255, 255)
    canny_thresholds: tuple[int, int] = (50, 150)
    strict_marker_count: bool = True
    corner_mode: Literal["centers", "centers_to_layout", "outer_corners"] = "centers"
    min_quad_area_px: int = 100_000


class AlignResult(BaseModel):
    """Result of a successful alignment run.

    Returned by aligner.align() and NormalizerService. No "status" field —
    success/absence of exception models success; failures raise DomainError.
    """
    marker_ids: list[int]
    source_points: list[tuple[float, float]]
    dst_points: list[tuple[float, float]]
    homography: list[list[float]]
    output_size: tuple[int, int]
    corner_mode: str
    warnings: list[str] = Field(default_factory=list)


class CropBatch(BaseModel):
    """Batch of cropped ROIs ready for CNN training or inference (ЛР3/ЛР4)."""
    template_version: str
    crops: list[dict] = Field(default_factory=list)
    normalized_size: tuple[int, int] | None = None


class FieldGrade(BaseModel):
    """Grading result for a single field (ЛР5 Checker output)."""
    field_id: str
    predicted: str | None = None
    confidence: float | None = None
    status: Literal["CORRECT", "INCORRECT", "NEEDS_REVIEW"]


class GradeResult(BaseModel):
    """Full grading result for one photo (ЛР5 DAG output)."""
    template_version: str
    photo_id: str
    fields: list[FieldGrade] = Field(default_factory=list)
    score: float = 0.0
    max_score: float = 0.0
    needs_review: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class JobStatus(BaseModel):
    """Status of an asynchronous job (used by /api/jobs/{job_id})."""
    job_id: str
    state: Literal["queued", "running", "done", "error"]
    progress: float = 0.0
    result: GradeResult | None = None
    error: dict | None = None


class Settings(BaseModel):
    """Unified settings for Web + CLI + API (SKILL v4 §11).

    Priority: CLI flag > Settings > default. Stored once and injected everywhere.
    """
    confidence_threshold: float = 0.80
    canny_thresholds: tuple[int, int] = (50, 150)
    corner_mode: Literal["centers", "centers_to_layout", "outer_corners"] = "centers"
    padding_px: int = 5
    output_size: tuple[int, int] = (2100, 2970)
    tflite_pool_size: int = 4
    batch_size: int = 1
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    learning_rate: float = 0.001
    epochs: int = 30
    train_batch_size: int = 32


class TrainingConfig(BaseModel):
    """CNN training configuration (ЛР4)."""
    learning_rate: float = 0.001
    epochs: int = 30
    batch_size: int = 32
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    label_encoding: Literal["one_hot", "integer"] = "integer"
    input_shape: tuple[int, int, int] = (64, 128, 1)
    num_classes: int = 1


class FieldRecognizer(Protocol):
    """Protocol for pluggable field recognisers (ЛР3/ЛР4).

    Implementations (Tesseract, neural net, TFLite) are swapped via config;
    the normalizer never imports them directly.
    """
    def predict(self, crops: list) -> list[tuple[str, float]]:
        ...


class AnswerKey(BaseModel):
    """Ground-truth answers for a template (separate file, loaded at grading time).

    field_id keys must match TemplateSpec.roi_fields[].id; mismatch raises
    AnswerKeyMismatchError.
    """
    template_version: str
    answers: dict[str, str] = Field(default_factory=dict)