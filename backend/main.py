"""
SIGMA FastAPI application entry point.
"""
from __future__ import annotations

import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path

# ── Path bootstrap ────────────────────────────────────────────────────────────
# Ensure the repo root (parent of backend/) is on sys.path so that
# `from ml.input.pipeline import ...` works regardless of which directory
# uvicorn is launched from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from db.init import init_db
from api import analysis_router, results_router, upload_router, visualizations_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """Startup / shutdown lifecycle for the FastAPI app."""
    # Ensure upload directory exists
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

    # Attempt DB connection — non-fatal if atlas URI not yet set
    await init_db()

    yield

    # Nothing to tear down for Motor; connections close automatically


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="SIGMA — Signal Intelligence & Guided Modulation Analysis API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(o) for o in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Routers
app.include_router(upload_router, prefix=f"{settings.API_V1_STR}/upload", tags=["Upload"])
app.include_router(analysis_router, prefix=f"{settings.API_V1_STR}/analysis", tags=["Analysis"])
app.include_router(results_router, prefix=f"{settings.API_V1_STR}/results", tags=["Results"])
app.include_router(visualizations_router, prefix=f"{settings.API_V1_STR}/visualizations", tags=["Visualizations"])


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    from db.init import is_db_connected

    return {
        "status": "ok",
        "project": settings.PROJECT_NAME,
        "db_connected": is_db_connected(),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
