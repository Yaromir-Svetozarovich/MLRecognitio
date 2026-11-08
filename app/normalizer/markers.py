"""Marker detection and validation (этап 2 конвейера нормализации).

Отвечает за поиск ArUco-маркеров, проверку количества/уникальности ID,
и построение исходных точек (source points) для гомографии.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.core.errors import (
    DuplicateMarkerIdsError,
    MarkerIdsMismatchError,
    MarkersIncompleteError,
    MarkersNotFoundError,
    UnexpectedMarkerCountError,
)

# Fixed order of corners used for consistent source/destination point mapping
CORNER_ORDER = ["TL", "TR", "BR", "BL"]


def get_dictionary(dictionary_id: int):
    """Get the ArUco predefined dictionary by ID.

    Wrapper around cv2.aruco.getPredefinedDictionary for testability.
    """
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def detect_markers(image, dictionary_id: int):
    """Detect ArUco markers in an image.

    Uses cv2.aruco.detectMarkers when available (OpenCV < 4.8), otherwise
    falls back to ArucoDetector (OpenCV >= 4.8). Both return the same
    (corners, ids, rejected) contract.

    Args:
        image: BGR or grayscale numpy array.
        dictionary_id: ArUco dictionary ID (e.g. cv2.aruco.DICT_4X4_50).

    Returns:
        (corners, ids) where corners is a list of 4x2 arrays and ids is
        a numpy array of marker IDs.
    """
    dictionary = get_dictionary(dictionary_id)
    # ТЗ требует cv2.aruco.detectMarkers; в OpenCV >=4.8 функция удалена,
    # используем эквивалентный ArucoDetector с тем же контрактом.
    if hasattr(cv2.aruco, "detectMarkers"):
        corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary)
    else:
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(image)
    return corners, ids


def validate_markers(ids, expected_ids) -> list[int]:
    """Validate detected markers against expectations.

    Checks:
      - at least one marker found
      - exactly 4 markers (MVP: one sheet in frame)
      - unique IDs
      - ID set matches expected_ids from TemplateSpec / AlignConfig

    Args:
        ids: raw ids output from detectMarkers.
        expected_ids: list of expected marker IDs.

    Returns:
        Flat list of the 4 validated marker IDs.

    Raises:
        MarkersNotFoundError, MarkersIncompleteError, UnexpectedMarkerCountError,
        DuplicateMarkerIdsError, MarkerIdsMismatchError.
    """
    if ids is None or len(ids) == 0:
        raise MarkersNotFoundError()
    flat = [int(x) for x in ids.flatten().tolist()]
    if len(flat) < 4:
        raise MarkersIncompleteError(details={"found": len(flat)})
    if len(flat) > 4:
        raise UnexpectedMarkerCountError(details={"found": len(flat)})
    if len(set(flat)) != len(flat):
        raise DuplicateMarkerIdsError(details={"ids": flat})
    if set(flat) != set(int(x) for x in expected_ids):
        raise MarkerIdsMismatchError(details={"found": flat, "expected": list(expected_ids)})
    return flat


def _ordered_ids(corner_by_id: dict) -> list[int]:
    """Map corner names (TL, TR, BR, BL) to marker IDs using corner_by_id.

    corner_by_id maps marker_id -> corner name; we invert it to get
    marker_id for each corner in CORNER_ORDER.
    """
    corner_to_id = {v: k for k, v in corner_by_id.items()}
    return [int(corner_to_id[c]) for c in CORNER_ORDER]


def marker_centers(corners, ids) -> dict[int, np.ndarray]:
    """Compute the centre point of each detected marker.

    Args:
        corners: list of corner arrays from detectMarkers.
        ids: marker ID array.

    Returns:
        dict mapping marker_id -> (x, y) centre as float32 array.
    """
    flat = [int(x) for x in ids.flatten().tolist()]
    result = {}
    for i, mid in enumerate(flat):
        pts = corners[i].reshape(4, 2)
        result[mid] = pts.mean(axis=0)
    return result


def build_src_points(corners, ids, corner_by_id: dict, corner_mode: str) -> np.ndarray:
    """Build source points (4, 2 float32) ordered TL, TR, BR, BL.

    Supports three corner modes (SKILL v4 §4):
      - centers / centers_to_layout: use marker centres as source points
      - outer_corners: use the outermost corner of each marker (farthest
        from the centroid of all marker centres)

    Args:
        corners: raw corners from detectMarkers.
        ids: raw ids from detectMarkers.
        corner_by_id: mapping marker_id -> corner name.
        corner_mode: one of "centers", "centers_to_layout", "outer_corners".

    Returns:
        (src_points, ordered_ids) where src_points is (4, 2) float32 array
        ordered TL, TR, BR, BL.
    """
    if corner_by_id is None:
        raise ValueError("corner_by_id is required")
    flat = [int(x) for x in ids.flatten().tolist()]
    id_to_corners = {flat[i]: corners[i] for i in range(len(flat))}
    ordered_ids = _ordered_ids(corner_by_id)
    centers = marker_centers(corners, ids)

    if corner_mode in ("centers", "centers_to_layout"):
        src = np.array([centers[mid] for mid in ordered_ids], dtype=np.float32)
    elif corner_mode == "outer_corners":
        all_centers = np.array([centers[mid] for mid in ordered_ids], dtype=np.float32)
        centroid = all_centers.mean(axis=0)
        src_list = []
        for mid in ordered_ids:
            pts = id_to_corners[mid].reshape(4, 2)
            dists = np.linalg.norm(pts - centroid, axis=1)
            idx = int(np.argmax(dists))
            src_list.append(pts[idx])
        src = np.array(src_list, dtype=np.float32)
    else:
        raise ValueError(f"Unknown corner_mode: {corner_mode}")
    return src, ordered_ids