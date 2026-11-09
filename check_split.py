import json
from pathlib import Path

# Проверим split_proof.json
spec_path = Path("data/dataset/split_proof.json")
print(f"Файл существует: {spec_path.exists()}")
print(f"Полный путь: {spec_path.absolute()}")

if spec_path.exists():
    with open(spec_path, encoding="utf-8") as f:
        proof = json.load(f)
    print(f"Всего записей: {len(proof)}")
    
    # Посчитаем по сплитам
    from collections import Counter
    splits = Counter(proof.values())
    print(f"Распределение: {dict(splits)}")
    
    # Покажем первые 5 записей
    for i, (k, v) in enumerate(proof.items()):
        if i < 5:
            print(f"  {k} -> {v}")
        else:
            break
else:
    print("ФАЙЛ НЕ НАЙДЕН!")
    print("Посмотрим, что есть в data/dataset:")
    dataset_dir = Path("data/dataset")
    if dataset_dir.exists():
        for p in dataset_dir.iterdir():
            print(f"  {p.name}")
    else:
        print("  папка data/dataset не существует")