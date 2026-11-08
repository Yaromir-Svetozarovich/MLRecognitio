"""FastAPI application entry point.

HTML-интерфейс (Web) — основной интерфейс системы.
- Jinja2Templates для рендеринга страниц
- StaticFiles для CSS/JS
- API роуты (/api/*) остаются для программного доступа и CLI
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.deps import get_config
from app.api.routes import align as align_router
from app.api.routes import extract as extract_router
from app.api.routes import templates as templates_router

app = FastAPI(title="Blank Aligner", version="5.0.0")

# Статика и шаблоны
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# API роуты
app.include_router(templates_router.router, prefix="/api/templates", tags=["templates"])
app.include_router(align_router.router, prefix="/api/align", tags=["align"])
app.include_router(extract_router.router, prefix="/api/extract", tags=["extract"])


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Главная страница."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/align", response_class=HTMLResponse)
def align_page(request: Request):
    """Страница нормализации фото."""
    return templates.TemplateResponse("align.html", {"request": request})


@app.get("/templates", response_class=HTMLResponse)
def templates_page(request: Request):
    """Страница списка шаблонов."""
    return templates.TemplateResponse("templates.html", {"request": request})


@app.get("/health")
def health():
    """Дetailed health endpoint returning config state."""
    config = get_config()
    return {"status": "healthy", "data_dir": str(config.data_dir)}