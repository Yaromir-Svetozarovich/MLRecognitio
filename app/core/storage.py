"""Storage abstraction (local FS now, S3/MinIO later)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LocalStorage:
    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def path(self, *parts: str) -> Path:
        p = self.base_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def save_json(self, relative: str, data: Any) -> Path:
        p = self.path(relative)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return p

    def load_json(self, relative: str) -> Any:
        p = self.base_dir / relative
        return json.loads(p.read_text(encoding="utf-8"))

    def save_bytes(self, relative: str, data: bytes) -> Path:
        p = self.path(relative)
        p.write_bytes(data)
        return p

    def exists(self, relative: str) -> bool:
        return (self.base_dir / relative).exists()