"""
GET /api/v1/visualizations/{signal_id}/waveform
GET /api/v1/visualizations/{signal_id}/fft
GET /api/v1/visualizations/{signal_id}/psd
GET /api/v1/visualizations/{signal_id}/spectrogram

Visualization endpoints backed entirely by the real DSP pipeline
(dsp/visualization.py).  No mock or generated data is returned —
every response is computed from the actual uploaded signal file.

All endpoints require that the signal was already uploaded via
POST /api/v1/upload and its signal_id is known.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from db.init import is_db_connected

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Shared helpers ─────────────────────────────────────────────────────────────

async def _resolve_storage_path(signal_id: str) -> str:
    """
    Resolve signal_id → storage_path via DB or in-memory upload store.
    Raises HTTP 404 if not found.
    """
    if is_db_connected():
        try:
            from beanie import PydanticObjectId  # noqa: PLC0415
            from db.models import Signal         # noqa: PLC0415

            doc = await Signal.get(PydanticObjectId(signal_id))
            if doc:
                return doc.storage_path
        except Exception:  # noqa: BLE001
            pass

    # Fall back to in-memory upload store
    from api.upload import _UPLOAD_STORE  # noqa: PLC0415

    path = _UPLOAD_STORE.get(signal_id)
    if not path:
        raise HTTPException(
            status_code=404,
            detail=f"Signal '{signal_id}' not found. Upload the file first.",
        )
    return path


def _load_iq_sync(storage_path: str):
    """
    Load IQ data from a signal file synchronously (runs in a thread).
    Returns (iq_2d, sample_rate) using ml.input.pipeline.
    """
    from ml.input.pipeline import process_file  # type: ignore[import]

    segments, meta = process_file(storage_path)
    if meta.validation_status == "ERROR" or len(segments) == 0:
        raise ValueError(
            f"Signal file could not be loaded: {meta.error_message or 'no segments'}"
        )

    # Reconstruct full [2, N] IQ from all segments
    iq_full = segments.reshape(2, -1)  # [2, M*L]
    sample_rate = meta.sample_rate or 1.0
    return iq_full, sample_rate


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/{signal_id}/waveform")
async def get_waveform(signal_id: str, max_points: int = 1000):
    """
    Return time-domain waveform data (I, Q, time axis) for a signal.

    The IQ data is loaded from disk and decimated to at most ``max_points``
    samples for efficient frontend rendering.  All values are computed from
    the real uploaded signal — no mock data.
    """
    storage_path = await _resolve_storage_path(signal_id)

    loop = asyncio.get_event_loop()
    try:
        iq, sample_rate = await loop.run_in_executor(
            None, _load_iq_sync, storage_path
        )
    except Exception as exc:
        logger.exception("IQ load failed for signal %s", signal_id)
        raise HTTPException(status_code=422, detail=f"Signal load error: {exc}") from exc

    from dsp.visualization import generate_waveform_data  # noqa: PLC0415

    data = generate_waveform_data(iq, sample_rate, max_points=max_points)
    return data


@router.get("/{signal_id}/fft")
async def get_fft(signal_id: str, max_points: int = 1000):
    """
    Return FFT spectrum data (frequency bins and magnitude in dB) for a signal.

    Computed via the real DSP FFT implementation (dsp/fft.py).
    No mock data.
    """
    storage_path = await _resolve_storage_path(signal_id)

    loop = asyncio.get_event_loop()
    try:
        iq, sample_rate = await loop.run_in_executor(
            None, _load_iq_sync, storage_path
        )
    except Exception as exc:
        logger.exception("IQ load failed for signal %s", signal_id)
        raise HTTPException(status_code=422, detail=f"Signal load error: {exc}") from exc

    from dsp.visualization import generate_fft_data  # noqa: PLC0415

    data = generate_fft_data(iq, sample_rate, max_points=max_points)
    return data


@router.get("/{signal_id}/psd")
async def get_psd(signal_id: str, max_points: int = 1000):
    """
    Return Power Spectral Density data (frequency bins and power in dBW/Hz).

    Computed via Welch's method (dsp/psd.py).  No mock data.
    """
    storage_path = await _resolve_storage_path(signal_id)

    loop = asyncio.get_event_loop()
    try:
        iq, sample_rate = await loop.run_in_executor(
            None, _load_iq_sync, storage_path
        )
    except Exception as exc:
        logger.exception("IQ load failed for signal %s", signal_id)
        raise HTTPException(status_code=422, detail=f"Signal load error: {exc}") from exc

    from dsp.visualization import generate_psd_data  # noqa: PLC0415

    data = generate_psd_data(iq, sample_rate, max_points=max_points)
    return data


@router.get("/{signal_id}/spectrogram")
async def get_spectrogram(
    signal_id: str,
    max_time_points: int = 200,
    max_freq_points: int = 200,
):
    """
    Return STFT spectrogram matrix data (time, frequency, power in dBW/Hz).

    Computed via scipy STFT (dsp/spectrogram.py).  The matrix is downsampled
    to at most ``max_time_points`` × ``max_freq_points`` for efficient transfer.
    No mock data.
    """
    storage_path = await _resolve_storage_path(signal_id)

    loop = asyncio.get_event_loop()
    try:
        iq, sample_rate = await loop.run_in_executor(
            None, _load_iq_sync, storage_path
        )
    except Exception as exc:
        logger.exception("IQ load failed for signal %s", signal_id)
        raise HTTPException(status_code=422, detail=f"Signal load error: {exc}") from exc

    from dsp.visualization import generate_spectrogram_data  # noqa: PLC0415

    data = generate_spectrogram_data(
        iq,
        sample_rate,
        max_time_points=max_time_points,
        max_freq_points=max_freq_points,
    )
    return data
