from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel, Field


class RoiSpec(BaseModel):
    field_id: str
    label: str
    kind: Literal["handwriting", "checkbox", "digits", "text"]
    x: int
    y: int
    w: int
    h: int


class TemplateSpec(BaseModel):
    template_id: str
    version: str
    dictionary_id: int
    expected_marker_ids: list[int]
    corner_by_id: dict[int, Literal["TL", "TR", "BR", "BL"]]
    marker_layout: dict[int, tuple[int, int]]
    marker_size_px: int
    canvas_size: tuple[int, int] = (2100, 2970)
    rois: list[RoiSpec] = Field(default_factory=list)
    created_at: datetime


@dataclass(frozen=True)
class AlignConfig:
    dictionary_id: int = 0  # cv2.aruco.DICT_4X4_50 == 0
    expected_ids: tuple[int, ...] = (0, 1, 2, 3)
    output_size: tuple[int, int] = (2100, 2970)
    border_value: tuple[int, int, int] = (255, 255, 255)
    canny_thresholds: tuple[int, int] = (50, 150)
    strict_marker_count: bool = True
    corner_mode: str = "outer_corners"
    min_quad_area_px: int = 100_000


@dataclass
class AlignmentResult:
    output_image: np.ndarray
    edges_image: np.ndarray | None
    marker_ids: list[int]
    source_points: np.ndarray
    homography: np.ndarray
    output_size: tuple[int, int]
    warnings: list[str] = field(default_factory=list)


class FieldResult(BaseModel):
    field_id: str
    value: str | None = None
    confidence: float | None = None


class RecognizeResult(BaseModel):
    template_id: str
    fields: list[FieldResult] = Field(default_factory=list)


class FieldRecognizer(Protocol):
    def recognize(self, image: np.ndarray, roi: RoiSpec, spec: TemplateSpec) -> FieldResult:
        ...