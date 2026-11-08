"""Service layer: I/O, saving artifacts, error mapping.

NormalizerService is the thin adapter between the pure CV node (aligner.py)
and the outside world (filesystem, API, CLI). The pipeline itself never
touches disk or stdout.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.core.config import AppConfig
from app.core.errors import ImageReadError, InvalidImageError, WarpError
from app.core.schemas import AlignConfig, AlignResult, TemplateSpec
from app.core.storage import LocalStorage
from app.normalizer import aligner


class NormalizerService:
    def __init__(self, config: AppConfig, storage: LocalStorage | None = None):
        self.config = config
        self.storage = storage or LocalStorage(config.data_dir)

    def _build_align_config(self, corner_mode: str | None = None) -> AlignConfig:
        """Build an AlignConfig from AppConfig, optionally overriding corner_mode."""
        cm = corner_mode or self.config.corner_mode
        return AlignConfig(
            dictionary_id=self.config.dictionary_id,
            output_size=self.config.canvas_size,
            canny_thresholds=self.config.canny_thresholds,
            corner_mode=cm,
            min_quad_area_px=100_000,
        )

    def read_image(self, source) -> np.ndarray:
        """Read an image from a file path, raw bytes, or numpy array.

        Used by both CLI and API entry points. Raises ImageReadError /
        InvalidImageError on failure (never returns None).
        """
        if isinstance(source, (str, Path)):
            img = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if img is None:
                raise ImageReadError(details={"path": str(source)})
            return img
        if isinstance(source, (bytes, bytearray)):
            arr = np.frombuffer(source, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise InvalidImageError()
            return img
        if isinstance(source, np.ndarray):
            return source
        raise InvalidImageError(details={"reason": "unsupported source"})

    def align(
        self,
        image: np.ndarray,
        template_spec: TemplateSpec | None = None,
        corner_mode: str | None = None,
        save_aligned_path: str | Path | None = None,
        save_edges_path: str | Path | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, AlignResult]:
        """Run alignment and optionally save artifacts to disk.

        The actual CV work is delegated to aligner.align(); this method only
        handles configuration building and file I/O.
        """
        image = self.read_image(image)
        cfg = self._build_align_config(corner_mode)
        output_image, edges_image, result = aligner.align(image, cfg, template_spec)

        sc = self.config.normalizer
        if save_aligned_path is not None and sc.save_aligned:
            p = Path(save_aligned_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(p), output_image)
            if not ok:
                raise WarpError(details={"reason": "failed to write aligned image"})
        if save_edges_path is not None and sc.save_edges:
            p = Path(save_edges_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            if edges_image is not None:
                cv2.imwrite(str(p), edges_image)
        return output_image, edges_image, result

    def align_file(
        self,
        input_path: str | Path,
        template_spec: TemplateSpec | None = None,
        corner_mode: str | None = None,
        save_aligned_path: str | Path | None = None,
        save_edges_path: str | Path | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, AlignResult]:
        """Convenience: read image from disk, then align."""
        image = self.read_image(input_path)
        return self.align(
            image,
            template_spec=template_spec,
            corner_mode=corner_mode,
            save_aligned_path=save_aligned_path,
            save_edges_path=save_edges_path,
        )