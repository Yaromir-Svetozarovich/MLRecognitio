"""Domain errors and their HTTP mapping."""

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
    def __init__(self, message: str = "Could not read image file", details=None):
        super().__init__("ImageReadError", message, 400, details)


class InvalidImageError(DomainError):
    def __init__(self, message: str = "Invalid or corrupt image", details=None):
        super().__init__("InvalidImageError", message, 400, details)


class UploadTooLargeError(DomainError):
    def __init__(self, message: str = "Upload exceeds size limit", details=None):
        super().__init__("UploadTooLargeError", message, 413, details)


class MarkersNotFoundError(DomainError):
    def __init__(self, message: str = "No ArUco markers found", details=None):
        super().__init__("MarkersNotFoundError", message, 422, details)


class MarkersIncompleteError(DomainError):
    def __init__(self, message: str = "Found fewer than 4 markers", details=None):
        super().__init__("MarkersIncompleteError", message, 422, details)


class UnexpectedMarkerCountError(DomainError):
    def __init__(self, message: str = "Found more than 4 markers", details=None):
        super().__init__("UnexpectedMarkerCountError", message, 422, details)


class DuplicateMarkerIdsError(DomainError):
    def __init__(self, message: str = "Duplicate marker IDs", details=None):
        super().__init__("DuplicateMarkerIdsError", message, 422, details)


class MarkerIdsMismatchError(DomainError):
    def __init__(self, message: str = "Marker ID set does not match template", details=None):
        super().__init__("MarkerIdsMismatchError", message, 422, details)


class TemplateSpecMismatchError(DomainError):
    def __init__(self, message: str = "Incompatible template spec version", details=None):
        super().__init__("TemplateSpecMismatchError", message, 422, details)


class HomographyError(DomainError):
    def __init__(self, message: str = "Degenerate homography", details=None):
        super().__init__("HomographyError", message, 500, details)


class WarpError(DomainError):
    def __init__(self, message: str = "Warp failed", details=None):
        super().__init__("WarpError", message, 500, details)


class ValidationError(DomainError):
    def __init__(self, message: str = "Validation failed", details=None):
        super().__init__("ValidationError", message, 400, details)