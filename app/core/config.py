"""Application configuration (loaded once, injected via Depends).

Central source of runtime settings. No os.getenv calls outside this module —
all configuration is read here via from_env() and passed to services through
FastAPI Depends or direct instantiation.

Settings (runtime, Pydantic) are stored in /data/settings.json, loaded at
startup, merged with defaults, and can be updated via PUT /api/settings or
CLI `settings`. Priority: CLI flag > Settings > default.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from app.core.schemas import Settings


@dataclass
class NormalizerServiceConfig:
    """I/O settings for the normalizer service layer.

    The pure pipeline (aligner.py) does not know about files; this config
    decides what gets saved to disk.
    """
    save_edges: bool = False
    save_input: bool = False
    save_aligned: bool = True
    save_crops: bool = False
    output_filenames: dict = None

    def __post_init__(self):
        if self.output_filenames is None:
            self.output_filenames = {"aligned": "aligned_blank.png", "edges": "edges.png"}


@dataclass
class AppConfig:
    """Global application configuration.

    Fields:
        data_dir:           root data directory (default "data")
        templates_dir:      where template specs and PNGs are stored
        jobs_dir:           where job artifacts (aligned.png, edges.png, result.json) go
        models_dir:         where trained models and TFLite exports are stored
        dictionary_id:      ArUco dictionary ID (cv2.aruco.DICT_4X4_50 == 0)
        canvas_size:        output canvas size in pixels (2100x2970 for A4 @10px/mm)
        max_upload_mb:      max upload size in megabytes (API guard before reading into memory)
        canny_thresholds:   Canny edge detector low/high thresholds
        corner_mode:        "centers" | "centers_to_layout" | "outer_corners"
        log_level:          Python logging level
        normalizer:         NormalizerServiceConfig (save flags, output filenames)
        default_settings_path: path to runtime Settings JSON file
    """
    data_dir: Path = Path("data")
    templates_dir: Path = Path("data/templates")
    jobs_dir: Path = Path("data/jobs")
    models_dir: Path = Path("data/models")
    dictionary_id: int = 0  # cv2.aruco.DICT_4X4_50
    canvas_size: tuple[int, int] = (2100, 2970)
    max_upload_mb: int = 10
    canny_thresholds: tuple[int, int] = (50, 150)
    corner_mode: str = "centers"
    log_level: str = "INFO"
    normalizer: NormalizerServiceConfig = field(default_factory=NormalizerServiceConfig)
    default_settings_path: Path = Path("data/settings.json")

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Build AppConfig from environment variables with sensible defaults."""
        def _int(name, default):
            v = os.getenv(name)
            return int(v) if v is not None else default

        data_dir = Path(os.getenv("DATA_DIR", "data"))
        return cls(
            data_dir=data_dir,
            templates_dir=Path(os.getenv("TEMPLATES_DIR", str(data_dir / "templates"))),
            jobs_dir=Path(os.getenv("JOBS_DIR", str(data_dir / "jobs"))),
            models_dir=Path(os.getenv("MODELS_DIR", str(data_dir / "models"))),
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
            corner_mode=os.getenv("CORNER_MODE", "centers"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            default_settings_path=Path(
                os.getenv("DEFAULT_SETTINGS_PATH", str(data_dir / "settings.json"))
            ),
        )


class SettingsManager:
    """Loads, merges, and persists runtime Settings (Pydantic) to/from disk.

    Settings live in /data/settings.json. On startup they are loaded and merged
    with Pydantic field defaults (file values override defaults). Updates are
    written back to the same file.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.path = Path(config.default_settings_path)
        self._settings: Settings | None = None

    def load(self) -> Settings:
        """Load Settings from disk, merging with Pydantic defaults.

        Missing keys in the JSON file fall back to Settings field defaults.
        Extra keys in the file are ignored.
        """
        if self._settings is not None:
            return self._settings

        defaults = Settings()  # Pydantic defaults
        data = {}

        if self.path.is_file():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    data = raw
            except (json.JSONDecodeError, OSError):
                data = {}

        # Merge: file values override defaults, missing keys keep defaults
        merged = defaults.model_dump()
        merged.update(data)
        self._settings = Settings.model_validate(merged)
        return self._settings

    def save(self, settings: Settings) -> None:
        """Persist Settings to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            settings.model_dump_json(indent=2), encoding="utf-8"
        )
        self._settings = settings

    def get(self) -> Settings:
        """Return the current Settings, loading on first access."""
        if self._settings is None:
            return self.load()
        return self._settings

    def reset(self) -> Settings:
        """Reset Settings to Pydantic defaults and persist."""
        defaults = Settings()
        self.save(defaults)
        return defaults