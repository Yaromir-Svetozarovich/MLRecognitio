"""Пакет обучения CNN (ЛР4).

Содержит:
  - config.py      — TrainingConfig (гиперпараметры обучения)
  - model.py       — CNN архитектура (Keras Functional API)
  - dataset_tf.py  — tf.data.Dataset конвейер с разделением по photo_id
  - metrics.py     — кастомная метрика Macro-F1
  - train_loop.py  — низкоуровневый цикл обучения на tf.GradientTape
  - train.py       — точка входа скрипта обучения

Правило: пакет не импортирует FastAPI. TensorFlow импортируется только внутри
модулей пакета (не в api/serving).
"""

from __future__ import annotations
