"""Domain errors and their HTTP mapping.

All domain exceptions inherit from DomainError and carry:
  - code:        short machine-readable error code (used in JSON responses)
  - message:     human-readable description
  - http_status: HTTP status code to return via FastAPI exception handler
  - details:     optional dict with extra context (e.g. found marker IDs)

Domains raise these; API/CLI layers translate them to responses / exit codes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DomainError(Exception):
    code: str
    message: str
    http_status: int = 500
    details: dict | None = None

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.code}: {self.message}"


class ImageReadError(DomainError):
    """Could not read the input image file (missing path, unreadable, etc.)."""
    def __init__(self, message: str = "Could not read image file", details=None):
        super().__init__("ImageReadError", message, 400, details)


class InvalidImageError(DomainError):
    """File is not a valid image / corrupt format / cv2.imdecode returned None."""
    def __init__(self, message: str = "Invalid or corrupt image", details=None):
        super().__init__("InvalidImageError", message, 400, details)


class UploadTooLargeError(DomainError):
    """Uploaded file exceeds AppConfig.max_upload_mb."""
    def __init__(self, message: str = "Upload exceeds size limit", details=None):
        super().__init__("UploadTooLargeError", message, 413, details)


class MarkersNotFoundError(DomainError):
    """No ArUco markers detected at all (bad lighting, occlusion, wrong dictionary)."""
    def __init__(self, message: str = "No ArUco markers found", details=None):
        super().__init__("MarkersNotFoundError", message, 422, details)


class MarkersIncompleteError(DomainError):
    """Found fewer than 4 markers — sheet cannot be reliably aligned."""
    def __init__(self, message: str = "Found fewer than 4 markers", details=None):
        super().__init__("MarkersIncompleteError", message, 422, details)


class UnexpectedMarkerCountError(DomainError):
    """Found more than 4 markers — MVP assumes exactly one sheet in frame."""
    def __init__(self, message: str = "Found more than 4 markers", details=None):
        super().__init__("UnexpectedMarkerCountError", message, 422, details)


class DuplicateMarkerIdsError(DomainError):
    """Two or more detected markers share the same ID."""
    def __init__(self, message: str = "Duplicate marker IDs", details=None):
        super().__init__("DuplicateMarkerIdsError", message, 422, details)


class MarkerIdsMismatchError(DomainError):
    """Detected marker ID set does not match expected_marker_ids from TemplateSpec."""
    def __init__(self, message: str = "Marker ID set does not match template", details=None):
        super().__init__("MarkerIdsMismatchError", message, 422, details)


class TemplateSpecMismatchError(DomainError):
    """TemplateSpec version is incompatible with current code / expectations."""
    def __init__(self, message: str = "Incompatible template spec version", details=None):
        super().__init__("TemplateSpecMismatchError", message, 422, details)


class HomographyError(DomainError):
    """Perspective transform matrix is degenerate (collinear points, zero area, NaN, rank < 3)."""
    def __init__(self, message: str = "Degenerate homography", details=None):
        super().__init__("HomographyError", message, 500, details)


class WarpError(DomainError):
    """cv2.warpPerspective failed or produced wrong-sized output."""
    def __init__(self, message: str = "Warp failed", details=None):
        super().__init__("WarpError", message, 500, details)


class ConfigError(DomainError):
    """Invalid configuration, e.g. centers_to_layout requested without TemplateSpec."""
    def __init__(self, message: str = "Configuration error", details=None):
        super().__init__("ConfigError", message, 422, details)


class ValidationError(DomainError):
    """Generic validation failure (used by template generator input validation)."""
    def __init__(self, message: str = "Validation failed", details=None):
        super().__init__("ValidationError", message, 400, details)


class CropOutOfBoundsError(DomainError):
    """Requested ROI crop lies outside the aligned image bounds."""
    def __init__(self, message: str = "Crop out of bounds", details=None):
        super().__init__("CropOutOfBoundsError", message, 400, details)


class JobNotFoundError(DomainError):
    """Requested job_id does not exist in storage."""
    def __init__(self, message: str = "Job not found", details=None):
        super().__init__("JobNotFoundError", message, 404, details)


class AnswerKeyMismatchError(DomainError):
    """AnswerKey field IDs do not match ROI field IDs."""
    def __init__(self, message: str = "Answer key mismatch", details=None):
        super().__init__("AnswerKeyMismatchError", message, 422, details)


class RoiExtractionError(DomainError):
    """ROI cropping / extraction failed for a specific field."""
    def __init__(self, message: str = "ROI extraction failed", details=None):
        super().__init__("RoiExtractionError", message, 422, details)


class TfliteLoadError(DomainError):
    """TFLite model file could not be loaded."""
    def __init__(self, message: str = "TFLite load failed", details=None):
        super().__init__("TfliteLoadError", message, 500, details)


class TfliteInferenceError(DomainError):
    """TFLite inference failed at runtime."""
    def __init__(self, message: str = "TFLite inference failed", details=None):
        super().__init__("TfliteInferenceError", message, 500, details)


class CheckerError(DomainError):
    """DAG Checker (CORRECT/INCORRECT/NEEDS_REVIEW) failed."""
    def __init__(self, message: str = "Checker failed", details=None):
        super().__init__("CheckerError", message, 500, details)


class DocumentAlignmentError(DomainError):
    """Document alignment failed at a higher level (wraps lower-level errors)."""
    def __init__(self, message: str = "Document alignment failed", details=None):
        super().__init__("DocumentAlignmentError", message, 422, details)