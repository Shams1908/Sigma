"""
POST /api/v1/upload

Accepts a WAV or raw IQ file, validates it via ml.input.pipeline.process_file,
persists the file on disk, creates a Signal document (if DB is live), and
returns the signal_id plus basic metadata.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

from api.schemas import SignalMetadata, UploadResponse
from core.config import settings
from db.init import is_db_connected

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory map signal_id → storage_path (fallback when DB is not connected)
_UPLOAD_STORE: dict[str, str] = {}


def _save_upload(upload: UploadFile) -> tuple[str, int]:
    """
    Synchronously write the uploaded file to the uploads directory.
    Returns (storage_path, file_size_bytes).
    Runs in a thread via run_in_executor so it doesn't block the event loop.
    """
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Preserve original extension; prefix with a UUID to avoid collisions
    suffix = Path(upload.filename or "signal").suffix or ".bin"
    filename_safe = f"{uuid.uuid4()}{suffix}"
    dest = upload_dir / filename_safe

    upload.file.seek(0)
    with open(dest, "wb") as fh:
        shutil.copyfileobj(upload.file, fh)

    file_size = os.path.getsize(dest)
    return str(dest), file_size


def _process_file_sync(path: str):
    """
    Calls ml.input.pipeline.process_file synchronously.
    Runs in a thread via run_in_executor.
    """
    from ml.input.pipeline import process_file  # type: ignore[import]

    return process_file(path)


@router.post("", response_model=UploadResponse, status_code=201)
async def upload_signal(file: UploadFile = File(...)):
    """
    Upload a signal file (WAV or raw IQ format).

    1. Save to disk
    2. Validate via ml.input.pipeline.process_file
    3. Create Signal document (skipped when DB is not yet configured)
    4. Return signal_id + metadata
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    loop = asyncio.get_event_loop()

    # --- Step 1: Save file to disk (blocking I/O off-thread) ---
    try:
        storage_path, file_size = await loop.run_in_executor(
            None, _save_upload, file
        )
    except Exception as exc:
        logger.exception("Failed to save uploaded file.")
        raise HTTPException(status_code=500, detail=f"File storage error: {exc}") from exc

    # --- Step 2: Validate with ml input pipeline (CPU-bound, off-thread) ---
    try:
        _segments, meta = await loop.run_in_executor(
            None, _process_file_sync, storage_path
        )
    except Exception as exc:
        # Clean up the saved file before returning error
        try:
            Path(storage_path).unlink(missing_ok=True)
        except OSError:
            pass
        logger.exception("ml.input.pipeline.process_file failed.")
        raise HTTPException(
            status_code=422,
            detail=f"File validation failed: {exc}",
        ) from exc

    if meta.validation_status == "ERROR":
        try:
            Path(storage_path).unlink(missing_ok=True)
        except OSError:
            pass
        raise HTTPException(
            status_code=422,
            detail=f"Signal validation error: {meta.error_message}",
        )

    # --- Step 3: Persist Signal document (best-effort when DB is live) ---
    signal_id = str(uuid.uuid4())
    ingestion_time = datetime.now(timezone.utc)

    if is_db_connected():
        try:
            from db.models import Signal  # noqa: PLC0415

            doc = Signal(
                id=None,
                filename=file.filename,
                detected_format=meta.detected_format,
                sample_rate=meta.sample_rate,
                storage_path=storage_path,
                uploaded_at=ingestion_time,
            )
            await doc.insert()
            signal_id = str(doc.id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DB insert failed (continuing without persistence): %s", exc)

    # --- Step 4: Build response matching frontend SignalMetadata ---
    sample_rate = meta.sample_rate or 0.0
    original_count = meta.original_sample_count or 0
    duration = (original_count / sample_rate) if sample_rate > 0 else 0.0

    response_meta = SignalMetadata(
        fileName=file.filename,
        sampleRate=sample_rate,
        centerFrequency=0.0,  # Carrier offset estimated in analysis phase
        duration=duration,
        fileSize=file_size,
        ingestionTime=ingestion_time.isoformat() + "Z",
    )

    # Always register path in memory so analysis can find it without DB
    _UPLOAD_STORE[signal_id] = storage_path

    return UploadResponse(signal_id=signal_id, metadata=response_meta)
