"""Geometry helpers."""

from __future__ import annotations

import numpy as np


CANVAS_PX_PER_MM = 10  # 2100 px / 210 mm


def mm_to_px(mm: float, px_per_mm: float = CANVAS_PX_PER_MM) -> int:
    return int(round(mm * px_per_mm))


def order_points_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).reshape(-1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)