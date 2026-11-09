"""Конфигурация обучения CNN (ЛР4).

Объединяет TrainingConfig из core/schemas.py с тренировочными путями и
параметрами вывода модели.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from app.core.schemas import TrainingConfig


@dataclass
class TrainConfig:
    """Полная конфигурация обучения.

    Поля из TrainingConfig (core/schemas.py) повторяются здесь для удобства
    чтения конфига обучения как единого объекта.
    """

    # --- Данные ---
    dataset_dir: Path  # директория с dataset/ (crops/, labels.csv, split_proof.json)
    output_dir: Path   # куда писать model.keras, best_weights, history.csv, confusion_matrix.png
    alphabet_path: Path = Path("alphabet.json")

    # --- Гиперпараметры (дублируют TrainingConfig для независимости) ---
    learning_rate: float = 0.001
    epochs: int = 30
    batch_size: int = 32
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    label_encoding: Literal["one_hot", "integer"] = "integer"
    input_shape: tuple[int, int, int] = (64, 128, 1)
    num_classes: int = 69

    # --- Логирование ---
    log_every: int = 1  # каждую эпоху выводить метрики
    save_best_only: bool = True

    def __post_init__(self):
        self.dataset_dir = Path(self.dataset_dir)
        self.output_dir = Path(self.output_dir)
        self.alphabet_path = Path(self.alphabet_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def to_training_config(self) -> TrainingConfig:
        """Преобразовать в TrainingConfig из core/schemas.py."""
        return TrainingConfig(
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            batch_size=self.batch_size,
            split_ratios=self.split_ratios,
            label_encoding=self.label_encoding,
            input_shape=self.input_shape,
            num_classes=self.num_classes,
        )
