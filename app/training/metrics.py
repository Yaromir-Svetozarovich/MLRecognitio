"""Кастомные метрики для обучения CNN (ЛР4).

Реализует Macro-F1 Score через tf.math.confusion_matrix, чтобы не
зависеть от tensorflow-addons. Метрика обновляется состоянием
(update_state / result / reset_states), как в Keras.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class MacroF1:
    """Кастомная метрика Macro-F1 Score.

    Считает F1 для каждого класса и усредняет без учёта поддержки
    (macro average). Работает через накопление confusion matrix.

    Использование:
        f1 = MacroF1(num_classes=N)
        f1.update_state(y_true, y_pred_logits)
        value = f1.result()   # скаляр
        f1.reset_states()
    """

    def __init__(self, num_classes: int, name: str = "macro_f1"):
        self.num_classes = num_classes
        self.name = name
        # confusion matrix накапливается как матрица NxN
        self._confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update_state(self, y_true, y_pred):
        """Обновить состояние метрики батчем данных.

        Args:
            y_true: массив истинных меток (int), форма (batch,).
            y_pred: массив предсказанных меток (int) или логитов (batch, num_classes).
        """
        import tensorflow as tf

        y_true = np.asarray(y_true).astype(np.int64).ravel()
        y_pred = np.asarray(y_pred)

        # Если пришли логиты —取 argmax
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1).astype(np.int64)
        else:
            y_pred = y_pred.astype(np.int64).ravel()

        # Накапливаем confusion matrix
        for t, p in zip(y_true, y_pred):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self._confusion[t, p] += 1

    def result(self) -> float:
        """Вычислить Macro-F1 на основе накопленной confusion matrix."""
        cm = self._confusion.astype(np.float64)

        # TP по диагонали
        tp = np.diag(cm)
        # FP: сумма по столбцу минус TP
        fp = cm.sum(axis=0) - tp
        # FN: сумма по строке минус TP
        fn = cm.sum(axis=1) - tp

        # F1 для каждого класса: 2*TP / (2*TP + FP + FN)
        denominator = 2 * tp + fp + fn
        f1_per_class = np.zeros(self.num_classes, dtype=np.float64)
        mask = denominator > 0
        f1_per_class[mask] = (2 * tp[mask]) / denominator[mask]

        # Macro-F1: среднее по классам
        return float(np.mean(f1_per_class))

    def reset_states(self):
        """Сбросить накопленное состояние."""
        self._confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def confusion_matrix(self) -> np.ndarray:
        """Вернуть копию накопленной confusion matrix."""
        return self._confusion.copy()


class AverageMeter:
    """Счётчик среднего значения (для потерь и точности за эпоху)."""

    def __init__(self, name: str = "meter"):
        self.name = name
        self._sum = 0.0
        self._count = 0

    def update(self, value: float, count: int = 1):
        self._sum += float(value) * count
        self._count += count

    def result(self) -> float:
        return self._sum / max(self._count, 1)

    def reset_states(self):
        self._sum = 0.0
        self._count = 0


def compute_confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    """Вычислить confusion matrix через tf.math.confusion_matrix."""
    import tensorflow as tf

    y_true = np.asarray(y_true).astype(np.int32).ravel()
    y_pred = np.asarray(y_pred)

    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1).astype(np.int32)
    else:
        y_pred = y_pred.astype(np.int32).ravel()

    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes)
    return cm.numpy()
