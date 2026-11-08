# ArUco Blank Generator

Генератор A4-бланков ответов с 4 ArUco-маркерами (DICT_4X4_50) и полями ответов.
Подготовлен к интеграции в проект нормализации/распознавания по SKILL (Blank Aligner).

## Структура

```
app/
  core/        схемы, конфиг, ошибки, геометрия, хранилище
  template/    generator.py (TemplateSpec), renderer.py (DOCX)
  cli.py       тонкий CLI-адаптер
tests/         pytest
```

## Запуск

```bat
setup.bat   # один раз: установка зависимостей
run.bat     # генерация blank_v1.docx / blank_v2.docx + spec.json
```

Или вручную:

```bash
python generate_blank.py --layout template_layout.json --content content_v1.json --output output\blank_v1.docx
```

## Контракты (SKILL)

- `TemplateSpec` — единственный источник истины о листе (corner_by_id, marker_layout, marker_size_px, canvas_size, roi_fields).
- `corner_by_id`: `0->TL`, `1->TR`, `2->BR`, `version_marker_id->BL` (каноническая раскладка SKILL v4).
- `marker_layout`: ID -> (x, y) центр маркера в пикселях холста 2100x2970.
- `marker_size_px`: физический размер маркера (15 мм = 150 px).
- Спека сохраняется рядом с DOCX: `output/<name>.spec.json`.
- `roi_fields` заполняются автоматически из `template_layout.json` (`roi.table_top_mm`, `roi.row_height_mm`, `question_columns_mm`). Меняя эти параметры, вы меняете и таблицу в DOCX, и координаты нарезки.

## Тесты

```bash
python -m pytest -q
```

## Примечания

- ROI-разметка полей ответов генерируется автоматически и соответствует таблице DOCX; координаты можно проверить в `output/<name>.spec.json` (`roi_fields`).
- Нормализатор и распознаватель не входят в этот репозиторий.