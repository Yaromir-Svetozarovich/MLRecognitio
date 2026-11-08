"""Preprocessing: grayscale + Canny edges (этап 1 конвейера нормализации).

Чистые функции, легко тестируются на синтетических изображениях.
"""

from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR/BGRA image to single-channel grayscale.

    Args:
        image: numpy array, HxW (already gray), HxWx3 (BGR) or HxWx4 (BGRA).

    Returns:
        HxW uint8 grayscale image.

    Raises:
        ValueError: if image is None or has unsupported channel count.
    """
    if image is None:
        raise ValueError("image is None")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def canny_edges(gray: np.ndarray, thresholds: tuple[int, int] = (50, 150)) -> np.ndarray:
    """Run Canny edge detector on a grayscale image.

    Args:
        gray: single-channel uint8 image.
        thresholds: (low, high) hysteresis thresholds.

    Returns:
        Binary edge map (uint8), same size as input.
    """
    low, high = thresholds
    return cv2.Canny(gray, low, high)


def preprocess(image: np.ndarray, thresholds: tuple[int, int] = (50, 150)):
    """Full preprocessing stage 1: BGR -> grayscale -> Canny edges.

    Returns:
        (gray, edges) tuple. The edges map is returned to the caller for
        visual evaluation; saving it to disk is the service/CLI/API concern,
        not the pipeline's.
    """
    gray = to_grayscale(image)
    edges = canny_edges(gray, thresholds)
    return gray, edges