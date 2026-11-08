"""Templates route: CRUD for template specs.

Endpoints:
  GET    /api/templates              list all stored templates
  GET    /api/templates/{id}         return TemplateSpec JSON
  GET    /api/templates/{id}/docx    download template DOCX for printing
  GET    /api/templates/{id}/png     download template PNG for printing (if available)
  GET    /api/templates/download/{path}  download individual generated file
  POST   /api/templates              upload spec.json + optional template_id
  POST   /api/templates/generate-work  create multi-variant work from CSV
"""

from __future__ import annotations

import io
import json
import random
import re
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse

from app.api.deps import get_config
from app.core.schemas import TemplateSpec
from app.core.storage import LocalStorage

router = APIRouter()


def _sanitize_folder_name(name: str) -> str:
    """Remove dangerous characters from a folder name."""
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = name.strip('. ')
    if not name:
        name = 'work'
    return name


@router.get("")
def list_templates():
    """Return a list of all templates stored in templates_dir."""
    config = get_config()
    storage = LocalStorage(config.templates_dir)
    templates = []
    if storage.base_dir.exists():
        for d in storage.base_dir.iterdir():
            spec_path = d / "spec.json"
            if spec_path.is_file():
                try:
                    data = json.loads(spec_path.read_text(encoding="utf-8"))
                    templates.append({"template_id": d.name, "spec": data})
                except Exception:
                    continue
    return JSONResponse({"templates": templates})


@router.get("/{template_id}")
def get_template(template_id: str):
    """Return the TemplateSpec JSON for a given template_id."""
    config = get_config()
    spec_path = config.templates_dir / template_id / "spec.json"
    if not spec_path.is_file():
        raise HTTPException(
            status_code=404,
            detail={"code": "JobNotFoundError", "message": "Template not found"},
        )
    data = json.loads(spec_path.read_text(encoding="utf-8"))
    return JSONResponse(data)


@router.get("/{template_id}/png")
def get_template_png(template_id: str):
    """Serve the printable template PNG for a given template_id."""
    config = get_config()
    png_path = config.templates_dir / template_id / "template.png"
    if not png_path.is_file():
        raise HTTPException(
            status_code=404,
            detail={"code": "JobNotFoundError", "message": "Template PNG not found"},
        )
    return FileResponse(str(png_path), media_type="image/png")


@router.get("/{template_id}/docx")
def get_template_docx(template_id: str):
    """Serve the printable template DOCX for a given template_id."""
    config = get_config()
    docx_path = config.templates_dir / template_id / "template.docx"
    if not docx_path.is_file():
        raise HTTPException(
            status_code=404,
            detail={"code": "JobNotFoundError", "message": "Template DOCX not found"},
        )
    return FileResponse(
        str(docx_path),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=f"{template_id}.docx",
    )


@router.get("/download/{file_path:path}")
def download_generated_file(file_path: str):
    """Serve an individual generated file (DOCX or ZIP) by its relative path under templates_dir or jobs_dir."""
    config = get_config()
    for base in (config.templates_dir, config.jobs_dir):
        candidate = base / file_path
        if candidate.is_file():
            if candidate.suffix.lower() == ".docx":
                media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            else:
                media_type = "application/zip"
            return FileResponse(
                str(candidate),
                media_type=media_type,
                filename=candidate.name,
            )
    raise HTTPException(
        status_code=404,
        detail={"code": "JobNotFoundError", "message": "Generated file not found"},
    )


@router.post("")
def create_template(
    spec_file: UploadFile = File(None),
    template_id: str = Form(None),
):
    """Upload a new template spec.json (and optional template_id override)."""
    config = get_config()
    storage = LocalStorage(config.templates_dir)

    if spec_file is None:
        raise HTTPException(
            status_code=400,
            detail={"code": "ValidationError", "message": "spec_file is required"},
        )

    spec_raw = spec_file.file.read()
    try:
        data = json.loads(spec_raw)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={"code": "ValidationError", "message": f"Invalid spec JSON: {e}"},
        )

    try:
        spec = TemplateSpec.model_validate(data)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={"code": "ValidationError", "message": f"Invalid TemplateSpec: {e}"},
        )

    tid = spec.template_version or template_id or "template_unknown"
    tdir = config.templates_dir / tid
    tdir.mkdir(parents=True, exist_ok=True)

    storage.save_bytes(f"{tid}/spec.json", spec.model_dump_json(indent=2).encode("utf-8"))

    return JSONResponse({
        "template_id": tid,
        "spec": spec.model_dump(),
        "message": "Template uploaded successfully",
    })
@router.post("/generate-work")
def generate_work(
    layout_file: UploadFile = File(...),
    questions_csv: UploadFile = File(...),
    work_name: str = Form(...),
    num_variants: int = Form(4),
    save_location: str = Form("templates"),
    copies_mode: str = Form("all"),
    copies_all: int = Form(1),
    version: int = Form(1),
    version_marker_id: int = Form(3),
    header_author: str = Form("Автор"),
    header_group: str = Form("Группа"),
    header_variant: str = Form("1"),
    header_course: str = Form("1"),
    instruction: str = Form(None),
):
    """Generate a multi-variant work from layout.json + questions.csv."""
    config = get_config()

    if num_variants < 1 or num_variants > 20:
        raise HTTPException(
            status_code=422,
            detail={"code": "ValidationError", "message": "num_variants must be 1..20"},
        )
    if version_marker_id < 3 or version_marker_id > 49:
        raise HTTPException(
            status_code=422,
            detail={"code": "ValidationError", "message": "version_marker_id must be 3..49"},
        )

    layout_raw = layout_file.file.read()
    try:
        layout = json.loads(layout_raw)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={"code": "ValidationError", "message": f"Invalid layout JSON: {e}"},
        )

    csv_raw = questions_csv.file.read()
    try:
        csv_text = csv_raw.decode("utf-8")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={"code": "ValidationError", "message": f"CSV is not valid UTF-8: {e}"},
        )

    from app.core.errors import DomainError
    from app.template.generator import (
        content_from_csv,
        build_template_spec,
        validate_inputs,
    )
    from app.template.renderer import generate_blank
    from docx import Document

    # URL-encode the path (in case of spaces or non-ASCII characters)
    from urllib.parse import quote

    if save_location == "jobs":
        base_dir = config.jobs_dir
    else:
        base_dir = config.templates_dir

    safe_work_name = _sanitize_folder_name(work_name)
    work_dir = base_dir / safe_work_name
    work_dir.mkdir(parents=True, exist_ok=True)

    generated_files = []
    for variant_idx in range(1, num_variants + 1):
        if copies_mode == "all":
            copies = copies_all
        else:
            copies = copies_all

        try:
            content = content_from_csv(
                csv_text=csv_text,
                header={
                    "author": work_name,
                    "group": header_group,
                    "variant": str(variant_idx),
                    "course": header_course,
                },
                instruction=instruction,
                version=version,
                version_marker_id=version_marker_id + variant_idx - 1,
            )
        except DomainError as e:
            raise HTTPException(
                status_code=422,
                detail={"code": e.code, "message": e.message, "details": e.details},
            )

        # Перемешиваем вопросы внутри каждого варианта, чтобы варианты
        # отличались (выборка из банка вопросов в случайном порядке).
        # Само содержимое content_from_csv уже валидно, здесь меняем только
        # порядок и заново нумеруем вопросы от 1 до N.
        variant_questions = list(content["questions"])
        random.shuffle(variant_questions)
        for idx, q in enumerate(variant_questions, 1):
            q["num"] = idx
        content["questions"] = variant_questions

        try:
            validate_inputs(layout, content)
        except DomainError as e:
            raise HTTPException(
                status_code=422,
                detail={"code": e.code, "message": e.message, "details": e.details},
            )

        try:
            spec = build_template_spec(layout, content, template_id=f"{safe_work_name}_variant_{variant_idx}")
        except DomainError as e:
            raise HTTPException(
                status_code=422,
                detail={"code": e.code, "message": e.message, "details": e.details},
            )

        # Защита от рассинхрона: число ROI-полей должно совпадать с числом вопросов.
        # Иначе набор полей ответа будет лишним или неполным.
        if len(spec.roi_fields) != len(content["questions"]):
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "RoiFieldCountMismatch",
                    "message": (
                        f"ROI fields ({len(spec.roi_fields)}) != "
                        f"questions ({len(content['questions'])})"
                    ),
                    "details": {
                        "roi_field_count": len(spec.roi_fields),
                        "question_count": len(content["questions"]),
                        "variant": variant_idx,
                    },
                },
            )

        variant_dir = work_dir / f"вариант_{variant_idx}"
        variant_dir.mkdir(parents=True, exist_ok=True)

        # Persist the TemplateSpec next to the variant so the normalizer and
        # ROI extractor can use the exact mapping/ROI geometry of this DOCX.
        spec_path = variant_dir / "spec.json"
        spec_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")

        for copy_idx in range(1, copies + 1):
            docx_filename = f"вариант_{variant_idx}_копия_{copy_idx}.docx"
            docx_path = variant_dir / docx_filename
            try:
                generate_blank(layout, content, str(docx_path))
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail={"code": "RenderError", "message": f"DOCX render failed: {e}"},
                )

            Document(str(docx_path))

            rel_path = str(docx_path.relative_to(base_dir))
            generated_files.append({
                "name": docx_filename,
                "path": rel_path,
                "variant": variant_idx,
                "copy": copy_idx,
            })

    manifest = {
        "work_name": work_name,
        "safe_name": safe_work_name,
        "num_variants": num_variants,
        "save_location": save_location,
        "copies_mode": copies_mode,
        "copies_all": copies_all,
        "version_marker_id_base": version_marker_id,
        "files": generated_files,
        "spec_files": [
            str(p.relative_to(base_dir))
            for p in sorted(work_dir.glob("вариант_*/spec.json"))
        ],
    }
    manifest_path = work_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in generated_files:
            abs_path = base_dir / f["path"]
            zf.write(abs_path, arcname=f["path"])
        for variant_spec in sorted(work_dir.glob("вариант_*/spec.json")):
            zf.write(
                variant_spec,
                arcname=str(variant_spec.relative_to(base_dir)),
            )
        zf.write(manifest_path, arcname=f"{safe_work_name}/manifest.json")
    zip_buffer.seek(0)

    zip_filename = f"{safe_work_name}.zip"
    zip_path = base_dir / safe_work_name / zip_filename
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path.write_bytes(zip_buffer.getvalue())

    zip_rel_path = f"{safe_work_name}/{zip_filename}"

    return JSONResponse({
        "work_name": work_name,
        "safe_name": safe_work_name,
        "files": generated_files,
        "zip_url": f"/api/templates/download/{quote(zip_rel_path, safe='/%')}",
        "message": f"Работа '{work_name}' создана: {num_variants} вариантов, {len(generated_files)} файлов",
    })