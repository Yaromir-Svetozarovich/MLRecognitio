"""Application configuration (loaded once, injected via Depends)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class NormalizerServiceConfig:
    save_edges: bool = False
    save_input: bool = False
    save_aligned: bool = True


@dataclass
class AppConfig:
    data_dir: Path = Path("data")
    templates_dir: Path = Path("data/templates")
    jobs_dir: Path = Path("data/jobs")
    dictionary_id: int = 0  # cv2.aruco.DICT_4X4_50
    canvas_size: tuple[int, int] = (2100, 2970)
    max_upload_mb: int = 10
    canny_thresholds: tuple[int, int] = (50, 150)
    corner_mode: str = "outer_corners"
    log_level: str = "INFO"
    normalizer: NormalizerServiceConfig = field(default_factory=NormalizerServiceConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        def _int(name, default):
            v = os.getenv(name)
            return int(v) if v is not None else default

        data_dir = Path(os.getenv("DATA_DIR", "data"))
        return cls(
            data_dir=data_dir,
            templates_dir=Path(os.getenv("TEMPLATES_DIR", str(data_dir / "templates"))),
            jobs_dir=Path(os.getenv("JOBS_DIR", str(data_dir / "jobs"))),
            dictionary_id=_int("DICTIONARY_ID", 0),
            canvas_size=(
                _int("CANVAS_WIDTH", 2100),
                _int("CANVAS_HEIGHT", 2970),
            ),
            max_upload_mb=_int("MAX_UPLOAD_MB", 10),
            canny_thresholds=(
                _int("CANNY_LOW", 50),
                _int("CANNY_HIGH", 150),
            ),
            corner_mode=os.getenv("CORNER_MODE", "outer_corners"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )