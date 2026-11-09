"""Импорт датасета constantinwerner/cyrillic-handwriting-dataset из Kaggle.

Ожидаемая структура после скачивания и распаковки:
    kaggle_root/
        train/
            aa1.png
            aa1007.png
            ...
        test/
            test0.png
            test1.png
            ...
        train.tsv     (image_name \t label) - кодировка UTF-8
        test.tsv      (image_name \t label) - кодировка UTF-8
"""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from app.dataset.splits import split_photo_ids


def write_labels_csv(
    output_dir: Path,
    split_files: Dict[str, List[Path]],
    labels_map: Dict[str, str],
) -> None:
    """Записать labels.csv с полями: crop_filename, label, photo_id, split."""
    import csv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_csv_path = output_dir / "labels.csv"
    with open(labels_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["crop_filename", "label", "photo_id", "split"])
        for split_name, files in split_files.items():
            for crop_file in files:
                label = labels_map.get(crop_file.name, "")
                photo_id = extract_photo_id(crop_file.name) or ""
                writer.writerow([crop_file.name, label, photo_id, split_name])


def write_split_proof(output_dir: Path, split_assignments: Dict[str, List[str]]) -> None:
    """Записать split_proof.json — отображение photo_id -> split."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    proof = {}
    for split_name, ids in split_assignments.items():
        for pid in ids:
            proof[pid] = split_name

    proof_path = output_dir / "split_proof.json"
    with open(proof_path, "w", encoding="utf-8") as f:
        json.dump(proof, f, ensure_ascii=False, indent=2)


def extract_photo_id(filename: str) -> str | None:
    """Извлечь photo_id из имени файла кропа.

    Ожидается формат: <photo_id>_<field_id>.png (например, photo_00001_А.png).
    """
    stem = Path(filename).stem
    if "_" not in stem:
        return None
    photo_id, _ = stem.rsplit("_", 1)
    return photo_id


def sanitize_filename_label(label: str) -> str:
    """Очистить метку для использования в имени файла.

    Заменяет недопустимые для Windows символы на подчёркивание.
    Оригинальная метка сохраняется в labels.csv без изменений.
    """
    import re

    # Заменяем всё, что не буква, цифра или подчёркивание
    safe = re.sub(r'[^\w\s-]', '_', label.strip())
    # Заменяем пробелы на подчёркивание
    safe = re.sub(r'\s+', '_', safe)
    # Убираем múltiровые подчёркивания
    safe = re.sub(r'_+', '_', safe)
    # Ограничиваем длину
    return safe[:50]


def import_kaggle_cyrillic(
    kaggle_root: str | Path,
    output_dir: str | Path,
    template_version: str = "v1",
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> dict:
    """Импортировать датасет constantinwerner/cyrillic-handwriting-dataset.

    Args:
        kaggle_root: путь к распакованному датасету из Kaggle.
        output_dir: куда записать train/, val/, test/, split_proof.json, labels.csv.
        template_version: версия шаблона для манифеста.
        ratios: доли train/val/test.
        seed: случайное зерно.

    Returns:
        Словарь с результатом импорта.

    Raises:
        FileNotFoundError: если не найден CSV с метками или папка с изображениями.
        ValueError: если не найдено изображений с метками.
    """
    kaggle_root = Path(kaggle_root)
    output_dir = Path(output_dir)

    # Читаем метки из TSV файлов (кодировка UTF-8)
    labels_map: Dict[str, str] = {}
    for tsv_name in ("train.tsv", "test.tsv"):
        tsv_path = kaggle_root / tsv_name
        if not tsv_path.is_file():
            continue
        with open(tsv_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 2:
                    image_name = row[0].strip()
                    label = row[1].strip()
                    if image_name and label:
                        labels_map[image_name] = label

    if not labels_map:
        raise ValueError(f"Не найдено пар image->label в TSV файлах {kaggle_root}")

    # Папки с изображениями
    train_dir = kaggle_root / "train"
    test_dir = kaggle_root / "test"

    # Копируем изображения в output_dir/crops/ с именами <photo_id>_<label>.png
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    photo_id_counter = 0
    grouped: Dict[str, List[Path]] = {}
    missing_labels = 0

    for img_dir in (train_dir, test_dir):
        if not img_dir.is_dir():
            continue
        for img_file in sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg")):
            label = labels_map.get(img_file.name)
            if label is None:
                missing_labels += 1
                continue

            photo_id = f"photo_{photo_id_counter:05d}"
            photo_id_counter += 1
            safe_label = sanitize_filename_label(label)
            new_name = f"{photo_id}_{safe_label}.png"
            dest = crops_dir / new_name
            if not dest.exists():
                shutil.copy2(str(img_file), str(dest))
            grouped.setdefault(photo_id, []).append(dest)

    if missing_labels:
        print(f"ПРЕДУПРЕЖДЕНИЕ: {missing_labels} изображений без меток пропущены")

    if not grouped:
        raise ValueError("Не найдено изображений с метками для импорта")

    # Сплит по photo_id
    split_photo_id_map = split_photo_ids(
        sorted(grouped.keys()), ratios=ratios, seed=seed
    )

    # photo_id -> split_name
    pid_to_split: Dict[str, str] = {}
    for split_name, pids in split_photo_id_map.items():
        for pid in pids:
            pid_to_split[pid] = split_name

    # Копируем файлы в подкаталоги сплитов
    split_files: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}
    for photo_id, files in grouped.items():
        split_name = pid_to_split.get(photo_id)
        if split_name is None:
            continue
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for crop_file in files:
            dest = split_dir / crop_file.name
            if not dest.exists():
                shutil.copy2(str(crop_file), str(dest))
            split_files[split_name].append(dest)

    # Пишем labels.csv
    write_labels_csv(output_dir, split_files, labels_map)

    # Пишем split_proof.json
    write_split_proof(output_dir, split_photo_id_map)

    # Пишем манифест
    manifest = {
        "template_version": template_version,
        "created_at": datetime.now().isoformat(),
        "source": "kaggle",
        "source_path": str(kaggle_root),
        "ratios": list(ratios),
        "seed": seed,
        "total_photo_ids": len(grouped),
        "total_crops": sum(len(v) for v in grouped.values()),
        "splits": {
            name: {
                "photo_ids": len(split_photo_id_map[name]),
                "crops": len(split_files[name]),
            }
            for name in ("train", "val", "test")
        },
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "status": "ok",
        "total_photo_ids": len(grouped),
        "total_crops": sum(len(v) for v in grouped.values()),
        "splits": {name: len(files) for name, files in split_files.items()},
        "output_dir": str(output_dir),
    }
