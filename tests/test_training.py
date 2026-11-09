"""Tests for the training package (ЛР4).

Тесты, требующие TensorFlow, пропускаются, если он не установлен
(код обучения живёт в отдельном образе trainer).
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.training.config import TrainConfig
from app.training.metrics import MacroF1, AverageMeter, compute_confusion_matrix


# ---------------------------------------------------------------------------
# MacroF1 (не требует TensorFlow)
# ---------------------------------------------------------------------------

class TestMacroF1:
    def test_perfect_predictions(self):
        """Идеальные предсказания дают F1=1.0."""
        f1 = MacroF1(num_classes=3)
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        f1.update_state(y_true, y_pred)
        assert f1.result() == 1.0

    def test_wrong_predictions(self):
        """Все неверные предсказания дают F1=0.0."""
        f1 = MacroF1(num_classes=3)
        y_true = np.array([0, 1, 2])
        y_pred = np.array([1, 2, 0])
        f1.update_state(y_true, y_pred)
        assert f1.result() == 0.0

    def test_partial(self):
        """Частичное совпадение дает промежуточное F1."""
        f1 = MacroF1(num_classes=3)
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 2])  # класс 0: 1/2, класс 1: 2/2, класс 2: 2/2
        f1.update_state(y_true, y_pred)
        result = f1.result()
        assert 0.0 < result < 1.0

    def test_reset_states(self):
        """После сброса метрика снова 0.0 (или не накоплена)."""
        f1 = MacroF1(num_classes=2)
        f1.update_state(np.array([0, 1]), np.array([0, 1]))
        assert f1.result() == 1.0
        f1.reset_states()
        f1.update_state(np.array([0]), np.array([1]))
        assert f1.result() == 0.0

    def test_logits_input(self):
        """Принимает логиты (2D массив) и делает argmax."""
        f1 = MacroF1(num_classes=3)
        y_true = np.array([0, 1, 2])
        y_pred_logits = np.array([
            [10.0, 0.0, 0.0],  # класс 0
            [0.0, 10.0, 0.0],  # класс 1
            [0.0, 0.0, 10.0],  # класс 2
        ])
        f1.update_state(y_true, y_pred_logits)
        assert f1.result() == 1.0


class TestAverageMeter:
    def test_basic(self):
        m = AverageMeter("test")
        m.update(0.5, count=2)
        m.update(1.0, count=1)
        assert m.result() == (0.5 * 2 + 1.0 * 1) / 3

    def test_reset(self):
        m = AverageMeter("test")
        m.update(1.0)
        assert m.result() == 1.0
        m.reset_states()
        assert m.result() == 0.0


def test_compute_confusion_matrix():
    """confusion matrix через tf.math.confusion_matrix."""
    from app.training.metrics import compute_confusion_matrix

    cm = compute_confusion_matrix(
        np.array([0, 1, 2, 0, 1]),
        np.array([0, 1, 1, 0, 2]),
        num_classes=3,
    )
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 2  # дважды предсказан 0 как 0
    assert cm[1, 2] == 1  # один раз 1 предсказан как 2


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------

class TestTrainConfig:
    def test_defaults(self):
        cfg = TrainConfig(dataset_dir=Path("/tmp/ds"), output_dir=Path("/tmp/out"))
        assert cfg.learning_rate == 0.001
        assert cfg.epochs == 30
        assert cfg.batch_size == 32
        assert cfg.input_shape == (64, 128, 1)
        assert cfg.num_classes == 69

    def test_to_training_config(self):
        cfg = TrainConfig(
            dataset_dir=Path("/tmp/ds"),
            output_dir=Path("/tmp/out"),
            learning_rate=0.01,
            epochs=10,
            num_classes=10,
        )
        tc = cfg.to_training_config()
        assert tc.learning_rate == 0.01
        assert tc.epochs == 10
        assert tc.num_classes == 10


# ---------------------------------------------------------------------------
# Alphabet
# ---------------------------------------------------------------------------

def test_alphabet_json_exists_and_valid():
    """alphabet.json существует и содержит 69 классов."""
    alphabet_path = Path("alphabet.json")
    assert alphabet_path.is_file(), "alphabet.json не найден в корне проекта"
    data = json.loads(alphabet_path.read_text(encoding="utf-8"))
    assert data["num_classes"] == 69
    assert len(data["classes"]) == 69
    assert len(data["class_to_index"]) == 69
    # Проверяем наличие русских и английских букв, цифр
    classes = set(data["classes"])
    assert "A" in classes and "Z" in classes
    assert "А" in classes and "Я" in classes
    assert "0" in classes and "9" in classes


# ---------------------------------------------------------------------------
# Model (требует TensorFlow)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    pytest.importorskip("tensorflow", reason="TensorFlow не установлен") is None,
    reason="TensorFlow не установлен",
)
class TestModel:
    def test_build_model(self):
        from app.training.model import build_model
        model = build_model(input_shape=(64, 128, 1), num_classes=69)
        assert model.output_shape == (None, 69)
        assert model.input_shape == (None, 64, 128, 1)

    def test_build_model_from_config(self):
        from app.training.model import build_model_from_config
        from app.core.schemas import TrainingConfig
        tc = TrainingConfig(num_classes=10, input_shape=(32, 64, 1))
        model = build_model_from_config(tc)
        assert model.output_shape == (None, 10)
