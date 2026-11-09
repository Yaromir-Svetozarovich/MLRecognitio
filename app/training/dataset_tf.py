"""tf.data.Dataset конвейер для обучения CNN (ЛР4).

Загружает изображения кропов и метки из каталога dataset/, создаёт
tf.data.Dataset с batch() и prefetch(AUTOTUNE), разделяет по photo_id
на train/val/test.

Правило: tensorflow импортируется только внутри функций.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np


def load_split_proof(dataset_dir: Path) -> dict:
    """Загрузить split_proof.json с отображением photo_id -> split."""
    proof_path = dataset_dir / "split_proof.json"
    if not proof_path.is_file():
        raise FileNotFoundError(f"split_proof.json не найден: {proof_path}")
    with open(proof_path, encoding="utf-8") as f:
        return json.load(f)


def load_labels(labels_path: Path) -> dict:
    """Загрузить labels.csv в словарь {crop_filename: label}."""
    import csv

    labels = {}
    with open(labels_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["crop_filename"]] = row["label"]
    return labels


def _parse_image(file_path: str, label: int, input_shape: Tuple[int, int, int]):
    """Прочитать одно изображение из файла, привести к гради и ресайз."""
    import tensorflow as tf

    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=1)  # grayscale
    img = tf.image.resize(img, [input_shape[0], input_shape[1]])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def make_tf_dataset(
    dataset_dir: Path,
    split: str = "train",
    batch_size: int = 32,
    input_shape: Tuple[int, int, int] = (64, 128, 1),
    shuffle: bool = True,
    seed: int = 42,
):
    """Создать tf.data.Dataset для данного сплита.

    Читает:
      - dataset_dir/crops/<photo_id>_<field_id>.png
      - dataset_dir/labels.csv (crop_filename, label, photo_id, split)
      - dataset_dir/split_proof.json (photo_id -> split)

    Возвращает tf.data.Dataset с batch() и prefetch(AUTOTUNE).

    Args:
        dataset_dir: корень датасета.
        split: "train", "val" или "test".
        batch_size: размер батча.
        input_shape: целевой размер изображения.
        shuffle: перемешивать ли данные (только для train).
        seed: случайное зерно для воспроизводимости.

    Returns:
        tf.data.Dataset, выдающий (images, labels) батчи.

    Raises:
        ImportError: если TensorFlow не установлен.
        FileNotFoundError: если split_proof.json или labels.csv отсутствуют.
    """
    import tensorflow as tf

    dataset_dir = Path(dataset_dir)

    # Загружаем split_proof: photo_id -> split
    split_proof = load_split_proof(dataset_dir)
    photo_ids_for_split = split_proof.get(split, [])
    if not photo_ids_for_split:
        raise ValueError(f"Нет photo_id для сплита '{split}' в split_proof.json")

    # Загружаем labels: crop_filename -> label
    labels_path = dataset_dir / "labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"labels.csv не найден: {labels_path}")
    labels_map = load_labels(labels_path)

    # Собираем файлы и метки для данного сплита
    crops_dir = dataset_dir / "crops"
    file_paths = []
    label_values = []
    label_to_index = {}  # будет заполнено из alphabet при необходимости

    # Собираем уникальные метки для кодирования
    unique_labels = sorted(set(labels_map.values()))
    label_to_index = {lbl: i for i, lbl in enumerate(unique_labels)}

    for photo_id in photo_ids_for_split:
        for crop_file in sorted(crops_dir.glob(f"{photo_id}_*.png")):
            crop_name = crop_file.name
            if crop_name in labels_map:
                file_paths.append(str(crop_file))
                label_values.append(label_to_index[labels_map[crop_name]])

    if not file_paths:
        raise ValueError(
            f"Не найдено изображений для сплита '{split}' с photo_id {photo_ids_for_split}"
        )

    # Создаём tf.data.Dataset
    paths_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    labels_ds = tf.data.Dataset.from_tensor_slices(
        tf.constant(label_values, dtype=tf.int32)
    )
    ds = tf.data.Dataset.zip((paths_ds, labels_ds))

    ds = ds.map(
        lambda path, label: _parse_image(path, label, input_shape),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_paths), seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds, len(unique_labels), label_to_index


def make_train_val_test_datasets(
    dataset_dir: Path,
    batch_size: int = 32,
    input_shape: Tuple[int, int, int] = (64, 128, 1),
    seed: int = 42,
):
    """Создать train, val, test tf.data.Dataset."""
    train_ds, num_classes, label_to_index = make_tf_dataset(
        dataset_dir, split="train", batch_size=batch_size,
        input_shape=input_shape, shuffle=True, seed=seed,
    )
    val_ds, _, _ = make_tf_dataset(
        dataset_dir, split="val", batch_size=batch_size,
        input_shape=input_shape, shuffle=False, seed=seed,
    )
    test_ds, _, _ = make_tf_dataset(
        dataset_dir, split="test", batch_size=batch_size,
        input_shape=input_shape, shuffle=False, seed=seed,
    )
    return train_ds, val_ds, test_ds, num_classes, label_to_index
