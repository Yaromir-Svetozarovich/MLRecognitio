"""CNN архитектура для распознавания рукописных символов (ЛР4).

Создаёт свёрточную сеть через Keras Functional API:

  Input (64, 128, 1) ->
    Conv2D(32, 3x3, relu) -> MaxPooling2D ->
    Conv2D(64, 3x3, relu) -> MaxPooling2D ->
    Conv2D(128, 3x3, relu) -> MaxPooling2D ->
    GlobalAveragePooling2D ->
    Dropout(0.5) ->
    Dense(num_classes, softmax)

Правило: tensorflow импортируется только внутри функций, чтобы пакет можно
было импортировать в окружении без TensorFlow (api, serving).
"""

from __future__ import annotations

from typing import Tuple

from app.core.schemas import TrainingConfig


def build_model(
    input_shape: Tuple[int, int, int] = (64, 128, 1),
    num_classes: int = 69,
):
    """Построить модель CNN для классификации рукописных символов.

    Архитектура (3 блока извлечения признаков + блок классификации):
      - 3x (Conv2D 3x3 relu + MaxPooling2D), фильтры 32 -> 64 -> 128
      - GlobalAveragePooling2D (GAP)
      - Dropout(0.5)
      - Dense(num_classes, softmax)

    Args:
        input_shape: форма входного тензора (height, width, channels).
        num_classes: количество классов в алфавите.

    Returns:
        Скомпилированная модель tf.keras.Model (без вызова model.fit).

    Raises:
        ImportError: если TensorFlow не установлен.
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "TensorFlow не установлен. Установите его в образе trainer: "
            "pip install tensorflow>=2.16,<3"
        ) from e

    inputs = keras.Input(shape=input_shape, name="input_image")

    # Блок 1: 32 фильтра
    x = layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same", name="conv1")(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # Блок 2: 64 фильтра
    x = layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same", name="conv2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    # Блок 3: 128 фильтров
    x = layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same", name="conv3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)

    # Блок классификации
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="handwritten_cnn")

    # Компиляция только для подсчёта потерь/метрик при обучении.
    # Обучение ведётся через низкоуровневый цикл (tf.GradientTape), model.fit не используется.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    return model


def build_model_from_config(config: TrainingConfig):
    """Построить модель из TrainingConfig."""
    return build_model(
        input_shape=config.input_shape,
        num_classes=config.num_classes,
    )
