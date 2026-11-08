"""API dependencies: config, storage, service injection.

Provides singleton-style accessors so FastAPI endpoints can inject shared
config, storage, and normalizer service via Depends.
"""

from __future__ import annotations

from app.core.config import AppConfig
from app.core.storage import LocalStorage
from app.normalizer.service import NormalizerService

_app_config: AppConfig | None = None
_normalizer_service: NormalizerService | None = None


def get_config() -> AppConfig:
    """Return the global AppConfig, loading from env on first call."""
    global _app_config
    if _app_config is None:
        _app_config = AppConfig.from_env()
    return _app_config


def get_storage(config: AppConfig) -> LocalStorage:
    """Return a LocalStorage bound to config.data_dir."""
    return LocalStorage(config.data_dir)


def get_normalizer_service(config: AppConfig) -> NormalizerService:
    """Return a singleton NormalizerService (created lazily)."""
    global _normalizer_service
    if _normalizer_service is None:
        storage = LocalStorage(config.data_dir)
        _normalizer_service = NormalizerService(config, storage)
    return _normalizer_service