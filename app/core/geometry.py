"""Geometry helpers for perspective transform computations.

Provides pure mathematical utilities with no OpenCV or domain dependencies,
making them easy to unit-test in isolation.
"""

from __future__ import annotations

import numpy as np

# 2100 px canvas width / 210 mm page width = 10 px per mm at 300 DPI equivalent
CANVAS_PX_PER_MM = 10


def mm_to_px(mm: float, px_per_mm: float = CANVAS_PX_PER_MM) -> int:
    """Convert millimetres to pixels using the canvas scale (10 px/mm by default)."""
    return int(round(mm * px_per_mm))


def order_points_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """Order 4 arbitrary points as TL, TR, BR, BL.

    Uses the classic sum/difference trick:
      - TL has the smallest x+y
      - BR has the largest x+y
      - TR has the smallest x-y
      - BL has the largest x-y

    Args:
        pts: array of shape (4, 2) with (x, y) coordinates.

    Returns:
        np.ndarray of shape (4, 2) ordered [TL, TR, BR, BL] as float32.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).reshape(-1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)