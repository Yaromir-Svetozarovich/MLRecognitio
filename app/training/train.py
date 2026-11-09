"""Точка входа скрипта обучения CNN (ЛР4).

Пример использования:
    python -m app.training.train \
        --dataset-dir data/dataset \
        --output-dir output/training \
        --alphabet alphabet.json \
        --epochs 30 \
        --batch-size 32 \
        --learning-rate 0.001 \
        --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.core.errors import DomainError
from app.training.config import TrainConfig
from app.training.dataset_tf import make_train_val_test_datasets, load_split_proof
from app.training.model import build_model_from_config
from app.training.train_loop import fit_loop, evaluate_on_test


def main(argv=None):
    parser = argparse.ArgumentParser(prog="app.training.train", help="Обучить CNN (ЛР4)")
    parser.add_argument("--dataset-dir", required=True, help="Директория датасета (с crops/, labels.csv, split_proof.json)")
    parser.add_argument("--output-dir", required=True, help="Куда сохранять модель и артефакты")
    parser.add_argument("--alphabet", default="alphabet.json", help="Путь к alphabet.json")
    parser.add_argument("--epochs", type=int, default=30, help="Количество эпох")
    parser.add_argument("--batch-size", type=int, default=32, help="Размер батча")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Скорость обучения")
    parser.add_argument("--input-height", type=int, default=64, help="Высота входного изображения")
    parser.add_argument("--input-width", type=int, default=128, help="Ширина входного изображения")
    parser.add_argument("--json", action="store_true", help="JSON-вывод")
    args = parser.parse_args(argv)

    try:
        alphabet_path = Path(args.alphabet)
        if not alphabet_path.is_file():
            print(f"ERROR: alphabet.json не найден: {alphabet_path}", file=sys.stderr)
            return 2

        with open(alphabet_path, encoding="utf-8") as f:
            alphabet = json.load(f)
        num_classes = alphabet["num_classes"]

        config = TrainConfig(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            alphabet_path=alphabet_path,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            input_shape=(args.input_height, args.input_width, 1),
            num_classes=num_classes,
        )

        # Проверяем split_proof.json
        split_proof = load_split_proof(config.dataset_dir)
        for split_name in ("train", "val", "test"):
            if split_name not in split_proof or not split_proof[split_name]:
                print(
                    f"ERROR: split_proof.json не содержит photo_id для сплита '{split_name}'",
                    file=sys.stderr,
                )
                return 2

        # Создаём tf.data.Dataset
        train_ds, val_ds, test_ds, ds_num_classes, label_to_index = (
            make_train_val_test_datasets(
                dataset_dir=config.dataset_dir,
                batch_size=config.batch_size,
                input_shape=config.input_shape,
                seed=42,
            )
        )

        if ds_num_classes != num_classes:
            print(
                f"WARNING: num_classes из alphabet ({num_classes}) != "
                f"из меток датасета ({ds_num_classes})",
                file=sys.stderr,
            )

        # Строим модель
        tc = config.to_training_config()
        model = build_model_from_config(tc)
        model.summary()

        # Обучаем
        history = fit_loop(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            output_dir=config.output_dir,
            num_classes=config.num_classes,
            label_to_index=label_to_index,
        )

        # Оцениваем на тесте
        test_results = evaluate_on_test(
            model=model,
            test_ds=test_ds,
            num_classes=config.num_classes,
            output_dir=config.output_dir,
            label_to_index=label_to_index,
        )

        if args.json:
            result = {
                "status": "ok",
                "epochs": config.epochs,
                "best_val_f1": max(history["val_f1"]),
                "test_loss": test_results["test_loss"],
                "test_macro_f1": test_results["test_macro_f1"],
                "artifacts": {
                    "model_keras": str(config.output_dir / "model.keras"),
                    "best_weights": str(config.output_dir / "best_weights.weights.h5"),
                    "history_csv": str(config.output_dir / "history.csv"),
                    "confusion_matrix_png": str(config.output_dir / "confusion_matrix.png"),
                    "confusion_matrix_csv": str(config.output_dir / "confusion_matrix.csv"),
                },
            }
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(
                f"\nОбучение завершено. Лучший val_f1={max(history['val_f1']):.4f}, "
                f"test_macro_f1={test_results['test_macro_f1']:.4f}"
            )
        return 0

    except DomainError as e:
        if args.json:
            print(json.dumps({"status": "error", "code": e.code, "message": e.message}))
        else:
            print(f"ERROR [{e.code}] {e.message}", file=sys.stderr)
        return 2
    except ImportError as e:
        if args.json:
            print(json.dumps({"status": "error", "code": "ImportError", "message": str(e)}))
        else:
            print(f"IMPORT ERROR: {e}", file=sys.stderr)
        return 3
    except Exception as e:  # noqa: BLE001
        if args.json:
            print(json.dumps({"status": "error", "code": "INTERNAL_ERROR", "message": str(e)}))
        else:
            print(f"ERROR: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
