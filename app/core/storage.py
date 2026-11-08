"""Storage abstraction layer.

LocalFilesystemStorage now; S3/MinIO later by implementing the same interface.
The domain code never calls open() / cv2.imwrite() directly — it goes through
storage, so swapping backends is a one-line change in service composition.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LocalStorage:
    """Filesystem-backed storage with automatic parent-directory creation."""

    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def path(self, *parts: str) -> Path:
        """Return absolute path under base_dir, creating parent dirs if needed."""
        p = self.base_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def save_json(self, relative: str, data: Any) -> Path:
        """Serialize data as JSON (utf-8, ensure_ascii=False) and write to disk."""
        p = self.path(relative)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return p

    def load_json(self, relative: str) -> Any:
        """Load and parse JSON from base_dir / relative."""
        p = self.base_dir / relative
        return json.loads(p.read_text(encoding="utf-8"))

    def save_bytes(self, relative: str, data: bytes) -> Path:
        """Write raw bytes to base_dir / relative."""
        p = self.path(relative)
        p.write_bytes(data)
        return p

    def exists(self, relative: str) -> bool:
        """Check whether base_dir / relative exists on disk."""
        return (self.base_dir / relative).exists()