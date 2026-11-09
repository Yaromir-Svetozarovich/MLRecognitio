"""build_dataset.py — разделение crops по photo_id и сборка датасета (ЛР4, Этап 1).

Скрипт принимает каталог с вырезанными ROI-изображениями, имена которых
содержат ID исходного бланка (например, blank001_roi15.png), и:

  1. Собирает все уникальные photo_id из имён файлов.
  2. Перемешивает photo_id и разбивает на train (70%), val (15%), test (15%).
  3. Распределяет файлы по подкаталогам train/, val/, test/.
  4. Пишет split_proof.json — отображение photo_id -> split.
  5. Пишет labels.csv — соответствие crop_filename -> label.

Использование:
    python build_dataset.py --crops-dir data/crops --labels data/labels.csv \
        --output-dir data/dataset --template-version v1 \
        --ratios 0.70 0.15 0.15 --seed 42
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def extract_photo_id(filename: str) -> str | None:
    """Извлечь photo_id из имени файла кропа.

    Ожидается формат: <photo_id>_<field_id>.png (например, blank001_roi15.png).
    photo_id — всё до последнего подчёркивания перед числом поля.

    Примеры:
        blank001_roi15.png -> blank001
        photo_42_q3_answer.png -> photo_42
    """
    stem = Path(filename).stem
    # Разделяем по последнему подчёркиванию: <photo_id>_<field>
    if "_" not in stem:
        return None
    photo_id, _ = stem.rsplit("_", 1)
    return photo_id


def scan_crops(crops_dir: Path) -> Dict[str, List[Path]]:
    """Собрать все файлы кропов, сгруппированные по photo_id.

    Возвращает словарь {photo_id: [путь_к_файлу, ...]}.
    """
    crops_dir = Path(crops_dir)
    if not crops_dir.is_dir():
        raise NotADirectoryError(f"Каталог с кропами не найден: {crops_dir}")

    grouped: Dict[str, List[Path]] = defaultdict(list)
    for crop_file in sorted(crops_dir.glob("*.png")):
        photo_id = extract_photo_id(crop_file.name)
        if photo_id is None:
            print(f"ПРЕДУПРЕЖДЕНИЕ: не удалось извлечь photo_id из {crop_file.name}, пропускаем",
                  file=sys.stderr)
            continue
        grouped[photo_id].append(crop_file)

    if not grouped:
        raise ValueError(f"Не найдено валидных файлов кропов в {crops_dir}")

    return grouped


def split_photo_ids(
    photo_ids: List[str],
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Разбить список photo_id на train/val/test строго по photo_id.

    Гарантирует, что все файлы одного photo_id попадают в один сплит
    (защита от утечки данных между train и test).
    """
    import random

    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")

    rng = random.Random(seed)
    shuffled = list(photo_ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def load_labels(labels_path: Path) -> Dict[str, str]:
    """Загрузить labels.csv в словарь {crop_filename: label}."""
    import csv

    labels = {}
    if not labels_path.is_file():
        return labels
    with open(labels_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["crop_filename"]] = row.get("label", "")
    return labels


def write_labels_csv(
    output_dir: Path,
    split_assignments: Dict[str, List[Path]],
    labels_map: Dict[str, str],
):
    """Записать labels.csv с полями: crop_filename, label, photo_id, split."""
    import csv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_csv_path = output_dir / "labels.csv"
    with open(labels_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["crop_filename", "label", "photo_id", "split"])
        for split_name, files in split_assignments.items():
            for crop_file in files:
                label = labels_map.get(crop_file.name, "")
                photo_id = extract_photo_id(crop_file.name) or ""
                writer.writerow([crop_file.name, label, photo_id, split_name])


def write_split_proof(output_dir: Path, split_assignments: Dict[str, List[str]]):
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


def copy_files_to_splits(
    grouped_by_photo_id: Dict[str, List[Path]],
    split_photo_id_map: Dict[str, List[str]],
    output_dir: Path,
) -> Dict[str, List[Path]]:
    """Скопировать файлы кропов в подкаталоги train/, val/, test/.

    Возвращает {split_name: [путь_к_скопированному_файлу, ...]}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # photo_id -> split_name
    pid_to_split = {}
    for split_name, pids in split_photo_id_map.items():
        for pid in pids:
            pid_to_split[pid] = split_name

    result: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}

    for photo_id, files in grouped_by_photo_id.items():
        split_name = pid_to_split.get(photo_id)
        if split_name is None:
            print(f"ПРЕДУПРЕЖДЕНИЕ: photo_id {photo_id} не попал ни в один сплит",
                  file=sys.stderr)
            continue

        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for crop_file in files:
            dest = split_dir / crop_file.name
            if not dest.exists():
                shutil.copy2(str(crop_file), str(dest))
            result[split_name].append(dest)

    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="build_dataset",
        description="Разделить ROI-крапы по photo_id на train/val/test (ЛР4, Этап 1)",
    )
    parser.add_argument("--crops-dir", required=True,
                        help="Каталог с файлами кропов (имя: <photo_id>_<field>.png)")
    parser.add_argument("--labels", default=None,
                        help="Путь к labels.csv (опционально, для меток классов)")
    parser.add_argument("--output-dir", required=True,
                        help="Куда записать train/, val/, test/, split_proof.json, labels.csv")
    parser.add_argument("--template-version", default="v1",
                        help="Версия шаблона (для манифеста)")
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.70, 0.15, 0.15],
                        help="Доли train/val/test (сумма должна быть 1.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Случайное зерно для воспроизводимости")
    parser.add_argument("--copy", action="store_true",
                        help="Копировать файлы вместо перемещения (по умолчанию копируем)")

    args = parser.parse_args(argv)

    crops_dir = Path(args.crops_dir)
    output_dir = Path(args.output_dir)
    labels_path = Path(args.labels) if args.labels else None
    ratios = tuple(args.ratios)

    # 1. Сканируем файлы кропов
    grouped = scan_crops(crops_dir)
    all_photo_ids = sorted(grouped.keys())
    print(f"Найдено {len(all_photo_ids)} уникальных photo_id, "
          f"{sum(len(v) for v in grouped.values())} файлов кропов")

    # 2. Разбиваем photo_id на сплиты
    split_photo_id_map = split_photo_ids(all_photo_ids, ratios=ratios, seed=args.seed)
    for split_name, pids in split_photo_id_map.items():
        print(f"  {split_name}: {len(pids)} photo_id")

    # 3. Копируем файлы в подкаталоги сплитов
    split_files = copy_files_to_splits(grouped, split_photo_id_map, output_dir)
    for split_name, files in split_files.items():
        print(f"  {split_name}: {len(files)} файлов скопировано")

    # 4. Загружаем метки (если есть)
    labels_map = {}
    if labels_path and labels_path.is_file():
        labels_map = load_labels(labels_path)
        print(f"Загружено {len(labels_map)} меток из {labels_path}")

    # 5. Пишем labels.csv
    write_labels_csv(output_dir, split_files, labels_map)

    # 6. Пишем split_proof.json
    write_split_proof(output_dir, split_photo_id_map)

    # 7. Пишем манифест датасета
    manifest = {
        "template_version": args.template_version,
        "created_at": datetime.now().isoformat(),
        "ratios": list(ratios),
        "seed": args.seed,
        "total_photo_ids": len(all_photo_ids),
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
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nГотово. Датасет собран в {output_dir}")
    print(f"  split_proof.json: {output_dir / 'split_proof.json'}")
    print(f"  labels.csv: {output_dir / 'labels.csv'}")
    print(f"  dataset_manifest.json: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
