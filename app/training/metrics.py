"""Кастомные метрики для обучения CNN (ЛР4).

Реализует Macro-F1 Score через tf.math.confusion_matrix, чтобы не
зависеть от tensorflow-addons. Метрика обновляется состоянием
(update_state / result / reset_states), как в Keras.
"""

from __future__ import annotations

import numpy as np


def _import_tf():
    """Импортировать TensorFlow с понятной ошибкой, если отсутствует."""
    try:
        import tensorflow as tf
        return tf
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "TensorFlow не установлен. Метрика Macro-F1 требует "
            "tensorflow>=2.16,<3 (образ trainer)."
        ) from e


class MacroF1:
    """Кастомная метрика Macro-F1 Score (graph-mode, tf.Variable).

    Считает F1 для каждого класса и усредняет без учёта поддержки
    (macro average). Состояние хранится в ``tf.Variable``, поэтому
    ``update_state`` можно вызывать внутри ``@tf.function``.

    Использование:
        f1 = MacroF1(num_classes=N)
        f1.update_state(y_true, y_pred_logits)   # numpy или tf.Tensor
        value = f1.result()                       # float (eager)
        f1.reset_states()
    """

    def __init__(self, num_classes: int, name: str = "macro_f1"):
        tf = _import_tf()
        self.num_classes = num_classes
        self.name = name
        # confusion matrix накапливается как tf.Variable (NxN)
        self._confusion = tf.Variable(
            tf.zeros((num_classes, num_classes), dtype=tf.int64),
            trainable=False,
            name=f"{name}_confusion",
        )

    def update_state(self, y_true, y_pred):
        """Обновить состояние метрики батчем данных (работает в graph mode).

        Args:
            y_true: истинные метки (int), форма (batch,).
            y_pred: предсказанные метки (int) или логиты (batch, num_classes).
        """
        tf = _import_tf()

        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.convert_to_tensor(y_pred)

        # Если пришли логиты — берём argmax
        if y_pred.shape.rank is not None and y_pred.shape.rank > 1:
            y_pred = tf.argmax(y_pred, axis=1, output_type=tf.int32)
        else:
            y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.int32)

        cm = tf.math.confusion_matrix(
            y_true, y_pred, num_classes=self.num_classes, dtype=tf.int64
        )
        self._confusion.assign_add(cm)

    def result(self) -> float:
        """Вычислить Macro-F1 на основе накопленной confusion matrix."""
        tf = _import_tf()

        cm = tf.cast(self._confusion, tf.float64)
        tp = tf.linalg.diag_part(cm)
        fp = tf.reduce_sum(cm, axis=0) - tp
        fn = tf.reduce_sum(cm, axis=1) - tp

        # F1 для каждого класса: 2*TP / (2*TP + FP + FN)
        denominator = 2 * tp + fp + fn
        f1_per_class = tf.where(
            denominator > 0,
            2 * tp / tf.maximum(denominator, 1e-12),
            tf.zeros_like(denominator),
        )

        # Macro-F1: среднее по классам
        return float(tf.reduce_mean(f1_per_class).numpy())

    def reset_states(self):
        """Сбросить накопленное состояние."""
        tf = _import_tf()
        self._confusion.assign(tf.zeros_like(self._confusion))

    def confusion_matrix(self) -> np.ndarray:
        """Вернуть копию накопленной confusion matrix."""
        return self._confusion.numpy().copy()


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
