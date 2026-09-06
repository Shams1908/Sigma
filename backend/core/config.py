"""
Application settings loaded from environment variables / .env file via pydantic-settings.
Drop a real MONGODB_URI into .env and the app connects with no code changes.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings

# Root of the repo (two levels up from this file: backend/core/ -> backend/ -> repo root)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    PROJECT_NAME: str = "SIGMA"
    API_V1_STR: str = "/api/v1"

    # MongoDB — read from .env; no default so misconfiguration is obvious
    MONGODB_URI: str = "mongodb+srv://abhiramsharmar_db_user:iKMWDBusc8Irb340@cluster0.clyc2ps.mongodb.net"
    DB_NAME: str = "AudioSIH"

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # Where uploaded files are stored (relative to repo root)
    UPLOAD_DIR: str = str(_REPO_ROOT / "uploads")

    # Path to the trained CNN model checkpoint (relative to repo root)
    MODEL_PATH: str = str(_REPO_ROOT / "models" / "m5_iq_cnn.pt")

    class Config:
        case_sensitive = True
        env_file = str(_REPO_ROOT / ".env")
        env_file_encoding = "utf-8"


settings = Settings()
