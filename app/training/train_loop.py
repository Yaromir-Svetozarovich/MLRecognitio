"""Низкоуровневый цикл обучения CNN (ЛР4).

Реализует пользовательский цикл обучения на базе tf.GradientTape с
разделением на train_step и val_step. Вызов model.fit ЗАПРЕЩЁН.

Структура:
  - train_step(images, labels, model, optimizer, loss_fn, metrics) -> dict
  - val_step(images, labels, model, loss_fn, metrics) -> dict
  - fit_loop(model, train_ds, val_ds, epochs, ...) -> history dict
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def _import_tf():
    """Импортировать TensorFlow с понятной ошибкой, если отсутствует."""
    try:
        import tensorflow as tf
        return tf
    except ImportError as e:
        raise ImportError(
            "TensorFlow не установлен. Обучающий конвейер требует "
            "tensorflow>=2.16,<3 (образ trainer)."
        ) from e


def train_step(
    images: np.ndarray,
    labels: np.ndarray,
    model,
    optimizer,
    loss_fn,
    train_metrics: Dict[str, Any],
) -> Dict[str, float]:
    """Один шаг обучения на батче через tf.GradientTape.

    Args:
        images: тензор изображений (batch, H, W, C).
        labels: тензор меток (batch,).
        model: tf.keras.Model.
        optimizer: tf.keras.optimizers.Optimizer.
        loss_fn: функция потерь (например, SparseCategoricalCrossentropy).
        train_metrics: словарь с объектом MacroF1 и AverageMeter для потерь.

    Returns:
        Словарь с loss, accuracy, macro_f1 для данного батча.
    """
    tf = _import_tf()

    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss = loss_fn(labels, logits)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Метрики
    predictions = tf.argmax(logits, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))

    # Macro-F1 обновляется состоянием
    f1_metric = train_metrics["macro_f1"]
    f1_metric.update_state(labels.numpy(), predictions.numpy())

    loss_meter = train_metrics["loss_meter"]
    loss_meter.update(float(loss), count=images.shape[0])

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "macro_f1": f1_metric.result(),
    }


def val_step(
    images: np.ndarray,
    labels: np.ndarray,
    model,
    loss_fn,
    val_metrics: Dict[str, Any],
) -> Dict[str, float]:
    """Один шаг валидации без градиентов."""
    tf = _import_tf()

    logits = model(images, training=False)
    loss = loss_fn(labels, logits)

    predictions = tf.argmax(logits, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))

    f1_metric = val_metrics["macro_f1"]
    f1_metric.update_state(labels.numpy(), predictions.numpy())

    loss_meter = val_metrics["loss_meter"]
    loss_meter.update(float(loss), count=images.shape[0])

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "macro_f1": f1_metric.result(),
    }


def fit_loop(
    model,
    train_ds,
    val_ds,
    epochs: int = 30,
    learning_rate: float = 0.001,
    output_dir: Path = Path("output/training"),
    num_classes: int = 69,
    label_to_index: dict | None = None,
) -> Dict[str, Any]:
    """Полный цикл обучения по эпохам с кастомным train_step/val_step.

    Сохраняет:
      - history.csv (Epoch, Train Loss, Val Loss, Train F1, Val F1)
      - best_weights при улучшении val_f1
      - confusion_matrix.png на тесте (после обучения)

    Args:
        model: tf.keras.Model.
        train_ds: tf.data.Dataset для обучения.
        val_ds: tf.data.Dataset для валидации.
        epochs: количество эпох.
        learning_rate: скорость обучения.
        output_dir: куда сохранять артефакты.
        num_classes: количество классов.
        label_to_index: отображение меток в индексы (для confusion matrix).

    Returns:
        Словарь с историей обучения.

    Raises:
        ImportError: если TensorFlow не установлен.
    """
    tf = _import_tf()
    from tensorflow import keras

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Метрики
    train_macro_f1 = _import_tf_metrics_macro_f1(num_classes)
    val_macro_f1 = _import_tf_metrics_macro_f1(num_classes)
    train_loss_meter = AverageMeter("train_loss")
    val_loss_meter = AverageMeter("val_loss")

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
    }

    best_val_f1 = -1.0
    best_weights_path = output_dir / "best_weights.weights.h5"

    for epoch in range(1, epochs + 1):
        # --- Обучение ---
        train_macro_f1.reset_states()
        train_loss_meter.reset_states()

        for images, labels in train_ds:
            train_step(images, labels, model, optimizer, loss_fn, {
                "macro_f1": train_macro_f1,
                "loss_meter": train_loss_meter,
            })

        train_loss = train_loss_meter.result()
        train_f1 = train_macro_f1.result()

        # --- Валидация ---
        val_macro_f1.reset_states()
        val_loss_meter.reset_states()

        for images, labels in val_ds:
            val_step(images, labels, model, loss_fn, {
                "macro_f1": val_macro_f1,
                "loss_meter": val_loss_meter,
            })

        val_loss = val_loss_meter.result()
        val_f1 = val_macro_f1.result()

        # --- Запись истории ---
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
        )

        # --- Чекпоинт по лучшему val_f1 ---
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_weights(str(best_weights_path))
            print(f"  -> Новый лучший val_f1={best_val_f1:.4f}, веса сохранены в {best_weights_path}")

    # Сохраняем историю в CSV
    history_csv_path = output_dir / "history.csv"
    with open(history_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Val Loss", "Train F1", "Val F1"])
        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i],
                f"{history['train_loss'][i]:.6f}",
                f"{history['val_loss'][i]:.6f}",
                f"{history['train_f1'][i]:.6f}",
                f"{history['val_f1'][i]:.6f}",
            ])
    print(f"История обучения сохранена в {history_csv_path}")

    # Сохраняем финальную модель
    model_keras_path = output_dir / "model.keras"
    model.save(str(model_keras_path))
    print(f"Модель сохранена в {model_keras_path}")

    return history


def _import_tf_metrics_macro_f1(num_classes: int):
    """Создать объект MacroF1 (из metrics.py)."""
    from app.training.metrics import MacroF1
    return MacroF1(num_classes=num_classes)


def evaluate_on_test(
    model,
    test_ds,
    num_classes: int,
    output_dir: Path = Path("output/training"),
    label_to_index: dict | None = None,
) -> Dict[str, Any]:
    """Оценить модель на тесте и построить confusion matrix."""
    tf = _import_tf()
    from app.training.metrics import MacroF1, compute_confusion_matrix
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_macro_f1 = MacroF1(num_classes=num_classes)
    test_loss_meter = AverageMeter("test_loss")
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    all_true = []
    all_pred = []

    for images, labels in test_ds:
        logits = model(images, training=False)
        loss = loss_fn(labels, logits)
        predictions = tf.argmax(logits, axis=1)

        test_macro_f1.update_state(labels.numpy(), predictions.numpy())
        test_loss_meter.update(float(loss), count=images.shape[0])

        all_true.extend(labels.numpy().tolist())
        all_pred.extend(predictions.numpy().tolist())

    test_loss = test_loss_meter.result()
    test_f1 = test_macro_f1.result()

    print(f"\nTest results: loss={test_loss:.4f}, macro_f1={test_f1:.4f}")

    # Confusion matrix
    cm = compute_confusion_matrix(
        np.array(all_true), np.array(all_pred), num_classes=num_classes
    )

    # Сохраняем confusion matrix как изображение
    cm_path = output_dir / "confusion_matrix.png"
    _plot_confusion_matrix(cm, num_classes, cm_path, label_to_index)
    print(f"Матрица ошибок сохранена в {cm_path}")

    # Сохраняем confusion matrix как CSV
    cm_csv_path = output_dir / "confusion_matrix.csv"
    np.savetxt(cm_csv_path, cm, fmt="%d", delimiter=",")
    print(f"Матрица ошибок (CSV) сохранена в {cm_csv_path}")

    return {
        "test_loss": float(test_loss),
        "test_macro_f1": float(test_f1),
        "confusion_matrix": cm.tolist(),
    }


def _plot_confusion_matrix(
    cm: np.ndarray,
    num_classes: int,
    output_path: Path,
    label_to_index: dict | None = None,
):
    """Нарисовать и сохранить матрицу ошибок."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(10, num_classes * 0.4), max(8, num_classes * 0.4)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Подписи осей
    if label_to_index:
        # Инвертируем для получения index -> label
        index_to_label = {v: k for k, v in label_to_index.items()}
        labels = [index_to_label.get(i, str(i)) for i in range(num_classes)]
    else:
        labels = [str(i) for i in range(num_classes)]

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Истинный класс",
        xlabel="Предсказанный класс",
        title="Матрица ошибок (Test)",
    )

    # Поворачиваем подписи по X для читаемости
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Подписываем значения в ячейках (только если не слишком много классов)
    if num_classes <= 30:
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=6,
                )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
