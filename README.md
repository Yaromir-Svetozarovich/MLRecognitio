# Blank Aligner v5

Система нормализации и распознавания A4-бланков ответов по ArUco-маркерам.

## Что делает

```
шаблон -> печать -> заполнение -> фото/скан ->
нормализация (выравнивание 2100×2970) -> нарезка ROI ->
обучение CNN (Custom Training Loop) -> TFLite ->
DAG: CV -> Extraction -> ML -> Checker -> GradeResult
```

## Статус по ЛР

| ЛР | Статус | Что сделано |
|----|--------|-------------|
| ЛР1 | ✅ Готово | Генератор DOCX-бланков с 4 ArUco-маркерами + TemplateSpec с авто-ROI |
| ЛР2 | ✅ Готово (код) | Нормализатор: фото -> aligned_blank.png + edges.png, API + Web UI |
| ЛР3 | ⚠️ Частично | Извлечение ROI-полей (полезная нагрузка), build-dataset, split по photo_id |
| ЛР4 | ⚠️ В разработке | Код обучения CNN готов, ждет TensorFlow + реальные данные |
| ЛР5 | ❌ Не начато | TFLite-экспорт, serving, DAG-оркестратор, Checker |

## Структура репозитория

```
app/
  core/           схемы (Pydantic), конфиг, ошибки, геометрия, хранилище
  template/       ГОТОВО: generator.py (TemplateSpec, resolve_marker_ids,
                  build_roi_fields), renderer.py (DOCX, плавающая таблица),
                  cli.py
  normalizer/     ЛР2 ГОТОВО: preprocessing.py, markers.py, aligner.py,
                  service.py (align принимает путь/bytes/ndarray)
  roi/            ЛР3 ГОТОВО: spec_loader.py, extractor.py
  dataset/        ЛР3-ЛР4 частично: builder.py, splits.py (photo_id 70/15/15)
  training/       ЛР4 в разработке: config.py, model.py (CNN), dataset_tf.py,
                  metrics.py (Macro-F1), train_loop.py (tf.GradientTape),
                  train.py (точка входа)
  api/            FastAPI: main.py, routes/{templates,align,extract}.py,
                  deps.py, static/, templates/
  cli.py          CLI-адаптер: align, extract, build-dataset, train
infra/            Docker: Dockerfile.api, Dockerfile.trainer, docker-compose.yml
tests/            pytest (тесты ЛР1–ЛР4)
alphabet.json     алфавит: 69 классов (A-Z, А-Я, 0-9) для рукописных символов
build_dataset.py  скрипт сплита датасета по photo_id
SKILL.txt         спецификация проекта (версия 5)
template_layout.json  разметка шаблона (маркеры, ROI, колонки)
content_v*.json   контент для генерации бланков
questions.csv     банк вопросов
```

## Быстрый старт

### 1. Установка зависимостей

```bat
setup.bat
```

### 2. Генерация бланка (ЛР1)

```bash
python generate_blank.py --layout template_layout.json --content content_v1.json --output output\blank_v1.docx --template-id template_v1_xxxx
```

Спека сохраняется рядом: `output\blank_v1.spec.json`.

### 3. Генерация работы из CSV (ЛР1/ЛР3)

Через CLI:
```bash
python -m app.cli ...  # (см. команды ниже)
```

Через Web: `http://localhost:8000/templates`

### 4. Нормализация фото (ЛР2)

Через Web: `http://localhost:8000/align`  
Загрузите фото бланка -> система найдет 4 ArUco-маркера и выровняет лист в 2100×2970.

После нормализации нажмите **"Извлечь поля ответов"** — система нарежет ROI-поля (q1_answer, q2_answer, ...) и покажет каждую как картинку.

### 5. Тесты

```bash
python -m pytest -q
```

Ожидаемый результат: все тесты проходят (без TensorFlow часть пропускается).

---

## CLI команды

```bash
# align — нормализация фото (ЛР2)
python -m app.cli align <input.jpg> --template <id|spec.json> --out aligned_blank.png --edges edges.png

# extract — нарезка ROI-полей (ЛР3)
python -m app.cli extract <aligned.png> --template <id|spec.json> --out crops/

# build-dataset — сборка датасета из кропов + меток (ЛР3)
python -m app.cli build-dataset --crops-dir <dir> --labels <labels.csv> --out dataset/ --template-version <ver> --photo-id <id>

# train — обучение CNN (ЛР4, требует TensorFlow)
python -m app.cli train --dataset-dir <dir> --output-dir <out> --alphabet alphabet.json --epochs 30 --batch-size 32

# build_dataset.py — автономный скрипт сплита по photo_id
python build_dataset.py --crops-dir <dir> --labels <labels.csv> --output-dir <out> --ratios 0.70 0.15 0.15 --seed 42
```

Exit codes: `0` — успех, `2` — доменная ошибка, `3` — I/O, `4` — ошибка обучения.

---

## Docker

### API (лёгкий, без TensorFlow)
```bash
docker compose -f infra/docker-compose.yml up -d api
# Веб-интерфейс: http://localhost:8000
```

### Trainer (с TensorFlow, оффлайн)
```bash
docker compose -f infra/docker-compose.yml --profile trainer up trainer
```

---

## API (FastAPI)

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/` | Главная страница Web UI |
| GET | `/align` | Страница нормализации фото |
| GET | `/templates` | Страница генерации бланков |
| GET | `/health` | Статус системы |
| POST | `/api/align` | Фото -> aligned + edges (data URL) |
| POST | `/api/extract` | Выровненное фото -> ROI-поля (data URL) |
| POST | `/api/templates/generate-work` | CSV + layout -> варианты DOCX + spec.json + ZIP |
| GET | `/api/templates/{id}/docx` | Скачать DOCX шаблона |

## Архитектурные решения (SKILL v5)

- **Каноническая раскладка маркеров:** `0 → TL`, `1 → TR`, `2 → BR`, `version_marker_id → BL`. Единственная точка истины — `resolve_marker_ids()` в `template/generator.py`.
- **Авто-ROI:** `roi_fields` заполняются автоматически из `template_layout.json` (`roi.table_top_mm`, `roi.row_height_mm`, `question_columns_mm`).
- **TemplateSpec** — единственный источник истины о листе.
- **Нет `model.fit`** — обучение через `tf.GradientTape` (train_step/val_step).
- **Нет TensorFlow в api/serving** — trainer lives only in offline image.
- **Сплит строго по photo_id** — защита от утечек данных между train/val/test.
- **Macro-F1** — кастомная метрика вместо простой Accuracy.

---


## Требования

- Python 3.11+
- opencv-contrib-python, fastapi, uvicorn, pydantic, jinja2, python-multipart, pytest, httpx
- TensorFlow >=2.16,<3 (только для обучения, образ trainer)

---

## Лицензия

MIT
