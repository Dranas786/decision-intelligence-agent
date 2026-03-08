from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.demo_routes import router as demo_router
from app.api.routes import router as api_router
from app.rag.rag_routes import router as rag_router


app = FastAPI(
    title="AI Decision Intelligence Agent",
    description="Agent that analyzes structured datasets and surfaces actionable insights.",
    version="0.1.0",
)


def _cors_allowed_origins() -> list[str]:
    configured = os.getenv("CORS_ALLOWED_ORIGINS", "*").strip()
    if not configured or configured == "*":
        return ["*"]
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(rag_router)
app.include_router(demo_router)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

INDEX_FILE = Path(__file__).resolve().parent / "static" / "index.html"


@app.get("/")
def root() -> FileResponse:
    return FileResponse(INDEX_FILE)


@app.get("/healthz")
def healthz() -> dict[str, str | bool]:
    qdrant_target = (
        os.getenv("QDRANT_URL", "").strip()
        or os.getenv("QDRANT_HOST", "").strip()
        or os.getenv("QDRANT_PATH", "data/qdrant")
    )
    return {
        "status": "ok",
        "frontend": "served_by_backend",
        "groq_configured": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "hash"),
        "qdrant_target": qdrant_target,
    }

