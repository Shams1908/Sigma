"""
POST /api/v1/analysis/{signal_id}
GET  /api/v1/analysis/{analysis_id}/status

Analysis creation launches the full hypothesis pipeline as a BackgroundTask
so the initiating request returns immediately with analysis_id.
The client polls the status endpoint until status becomes "done" or "failed".
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.schemas import AnalysisCreateResponse, AnalysisStatusResponse
from db.init import is_db_connected

logger = logging.getLogger(__name__)
router = APIRouter()

# ── In-memory fallback store (used when DB is not connected) ──────────────────
# Maps analysis_id → dict with status, signal_id, timestamps, etc.
_ANALYSIS_STORE: dict[str, dict[str, Any]] = {}


# ── Background analysis task ──────────────────────────────────────────────────

async def _run_analysis(analysis_id: str, signal_id: str, storage_path: str) -> None:
    """
    Background task: run the full hypothesis pipeline and persist results.

    Errors are caught at every level — the analysis always resolves to
    "done" or "failed", never stays "running" indefinitely.
    """
    timeout_seconds = 300  # 5-minute hard cap

    try:
        await asyncio.wait_for(
            _pipeline_and_persist(analysis_id, signal_id, storage_path),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        _set_failed(analysis_id, "Analysis timed out after 5 minutes.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in analysis task %s", analysis_id)
        _set_failed(analysis_id, f"Unexpected error: {exc}")


async def _pipeline_and_persist(
    analysis_id: str, signal_id: str, storage_path: str
) -> None:
    """Inner pipeline execution — separated so the timeout wraps it cleanly."""
    # Mark as running
    await _update_status(analysis_id, "running")

    # Run hypothesis pipeline in a thread (all CPU/IO-bound work)
    loop = asyncio.get_event_loop()
    try:
        from hypothesis.generator import run_hypothesis_pipeline  # noqa: PLC0415
        from core.config import settings  # noqa: PLC0415

        result = await loop.run_in_executor(
            None,
            lambda: run_hypothesis_pipeline(
                storage_path=storage_path,
                model_path=settings.MODEL_PATH,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        _set_failed(analysis_id, f"Pipeline execution failed: {exc}")
        return

    # Build parameter and hypothesis structures
    from db.models import ParameterEstimate, Hypothesis  # noqa: PLC0415

    params = ParameterEstimate(
        snr=result.snr,
        carrier_offset=result.carrier_offset,
        bandwidth=result.bandwidth,
        symbol_rate_estimate=result.symbol_rate_estimate,
    )

    hypotheses = []
    for h in result.ranked_hypotheses:
        try:
            hypotheses.append(
                Hypothesis(
                    modulation=h["modulation"],
                    symbol_rate=h["symbol_rate"],
                    fec_type=h.get("fec_type"),
                    ml_confidence=h["ml_confidence"],
                    calibrated_confidence=h.get("calibrated_confidence"),
                    sync_pass=h["sync_pass"],
                    demod_pass=h["demod_pass"],
                    fec_pass=h["fec_pass"],
                    bitstream_pass=h["bitstream_pass"],
                    final_score=h["final_score"],
                    rank=h["rank"],
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping malformed hypothesis entry: %s", exc)

    completed_at = datetime.utcnow()
    status = "done"
    error_msg = None

    if result.error_notes and not result.any_validated and not hypotheses:
        status = "failed"
        error_msg = "; ".join(result.error_notes[:3])

    # Persist to DB or in-memory store
    if is_db_connected():
        try:
            from beanie import PydanticObjectId  # noqa: PLC0415
            from db.models import Analysis       # noqa: PLC0415

            doc = await Analysis.get(PydanticObjectId(analysis_id))
            if doc:
                doc.status = status
                doc.completed_at = completed_at
                doc.parameters = params
                doc.hypotheses = hypotheses
                doc.error_message = error_msg
                await doc.save()
            else:
                logger.warning("Analysis doc %s not found in DB during update.", analysis_id)
                _update_memory_store(analysis_id, status, params, hypotheses, error_msg, completed_at)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DB persist failed for %s: %s", analysis_id, exc)
            _update_memory_store(analysis_id, status, params, hypotheses, error_msg, completed_at)
    else:
        _update_memory_store(analysis_id, status, params, hypotheses, error_msg, completed_at)


def _update_memory_store(
    analysis_id: str,
    status: str,
    params: Any,
    hypotheses: list,
    error_msg: str | None,
    completed_at: datetime,
) -> None:
    entry = _ANALYSIS_STORE.get(analysis_id, {})
    entry.update(
        {
            "status": status,
            "completed_at": completed_at,
            "parameters": params,
            "hypotheses": hypotheses,
            "error_message": error_msg,
        }
    )
    _ANALYSIS_STORE[analysis_id] = entry


async def _update_status(analysis_id: str, status: str) -> None:
    """Update just the status field."""
    if is_db_connected():
        try:
            from beanie import PydanticObjectId  # noqa: PLC0415
            from db.models import Analysis       # noqa: PLC0415

            doc = await Analysis.get(PydanticObjectId(analysis_id))
            if doc:
                doc.status = status
                await doc.save()
                return
        except Exception:  # noqa: BLE001
            pass

    if analysis_id in _ANALYSIS_STORE:
        _ANALYSIS_STORE[analysis_id]["status"] = status


def _set_failed(analysis_id: str, message: str) -> None:
    """Mark an analysis as failed synchronously (safe to call from exception handlers)."""
    logger.error("Analysis %s failed: %s", analysis_id, message)
    entry = _ANALYSIS_STORE.get(analysis_id, {})
    entry.update(
        {
            "status": "failed",
            "completed_at": datetime.utcnow(),
            "error_message": message,
        }
    )
    _ANALYSIS_STORE[analysis_id] = entry

    if is_db_connected():
        # Fire-and-forget coroutine for DB update — best-effort
        async def _db_fail():
            try:
                from beanie import PydanticObjectId  # noqa: PLC0415
                from db.models import Analysis       # noqa: PLC0415

                doc = await Analysis.get(PydanticObjectId(analysis_id))
                if doc:
                    doc.status = "failed"
                    doc.completed_at = datetime.utcnow()
                    doc.error_message = message
                    await doc.save()
            except Exception:  # noqa: BLE001
                pass

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_db_fail())
        except RuntimeError:
            pass


# ── Helpers: resolve signal_id → storage path ─────────────────────────────────

async def _get_storage_path(signal_id: str) -> str | None:
    """Look up the storage path for a signal_id in DB or in-memory upload store."""
    if is_db_connected():
        try:
            from beanie import PydanticObjectId  # noqa: PLC0415
            from db.models import Signal         # noqa: PLC0415

            doc = await Signal.get(PydanticObjectId(signal_id))
            if doc:
                return doc.storage_path
        except Exception:  # noqa: BLE001
            pass

    # Fall back to in-memory upload store imported from upload module
    from api.upload import _UPLOAD_STORE  # noqa: PLC0415

    return _UPLOAD_STORE.get(signal_id)


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/{signal_id}", response_model=AnalysisCreateResponse, status_code=202)
async def create_analysis(signal_id: str, background_tasks: BackgroundTasks):
    """
    Trigger analysis for an uploaded signal.

    Creates an Analysis record (pending), launches the hypothesis pipeline
    as a background task, and returns the analysis_id immediately.
    The client polls GET /analysis/{analysis_id}/status.
    """
    storage_path = await _get_storage_path(signal_id)
    if storage_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Signal '{signal_id}' not found. Upload the file first.",
        )

    analysis_id = str(uuid.uuid4())
    created_at = datetime.utcnow()

    if is_db_connected():
        try:
            from beanie import PydanticObjectId  # noqa: PLC0415
            from db.models import Analysis       # noqa: PLC0415

            doc = Analysis(
                signal_id=PydanticObjectId(signal_id),
                status="pending",
                created_at=created_at,
            )
            await doc.insert()
            analysis_id = str(doc.id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DB Analysis insert failed; using in-memory: %s", exc)

    # Always register in memory as a safety net
    _ANALYSIS_STORE[analysis_id] = {
        "signal_id": signal_id,
        "status": "pending",
        "created_at": created_at,
        "completed_at": None,
        "parameters": None,
        "hypotheses": [],
        "error_message": None,
        "storage_path": storage_path,
    }

    background_tasks.add_task(_run_analysis, analysis_id, signal_id, storage_path)

    return AnalysisCreateResponse(analysis_id=analysis_id, status="pending")


@router.get("/{analysis_id}/status", response_model=AnalysisStatusResponse)
async def get_analysis_status(analysis_id: str):
    """
    Poll the current status of an analysis job.

    Returns status ∈ {"pending", "running", "done", "failed"}.
    """
    # Try DB first
    if is_db_connected():
        try:
            from beanie import PydanticObjectId  # noqa: PLC0415
            from db.models import Analysis       # noqa: PLC0415

            doc = await Analysis.get(PydanticObjectId(analysis_id))
            if doc:
                return AnalysisStatusResponse(
                    analysis_id=analysis_id,
                    signal_id=str(doc.signal_id),
                    status=doc.status,
                    created_at=doc.created_at,
                    completed_at=doc.completed_at,
                    error_message=doc.error_message,
                )
        except Exception:  # noqa: BLE001
            pass

    # Fall back to in-memory
    entry = _ANALYSIS_STORE.get(analysis_id)
    if not entry:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis '{analysis_id}' not found.",
        )

    return AnalysisStatusResponse(
        analysis_id=analysis_id,
        signal_id=entry.get("signal_id", ""),
        status=entry.get("status", "pending"),
        created_at=entry.get("created_at", datetime.utcnow()),
        completed_at=entry.get("completed_at"),
        error_message=entry.get("error_message"),
    )


@router.get("/{signal_id}/visualizations")
async def get_visualizations(signal_id: str):
    """
    Return mock visualization data (spectrum, waterfall, constellation)
    for frontend display. This is temporary mock data for MVP.
    """
    import math
    import random
    
    # Mock spectrum data (FFT-like)
    spectrum = []
    for i in range(512):
        freq = (i - 256) / 512.0 * 2.4e6  # ±1.2 MHz around center
        # Gaussian peak at center + noise floor
        signal_component = 40 * math.exp(-(freq ** 2) / (2 * (200000 ** 2)))
        noise = random.uniform(-80, -70)
        magnitude = signal_component + noise
        spectrum.append({"frequency": freq, "magnitudeDb": magnitude})
    
    # Mock waterfall data (time-frequency matrix)
    waterfall = []
    for t in range(100):  # 100 time slices
        row = []
        for f in range(256):  # 256 frequency bins
            # Create a signal at center with some drift
            center_offset = math.sin(t * 0.1) * 20
            distance = abs(f - 128 - center_offset)
            signal_val = 40 * math.exp(-(distance ** 2) / 200) if distance < 50 else random.uniform(-80, -70)
            row.append(signal_val)
        waterfall.append(row)
    
    # Mock constellation data (QPSK-like)
    constellation = []
    points = [
        (0.7, 0.7), (-0.7, 0.7), (-0.7, -0.7), (0.7, -0.7)  # QPSK ideal points
    ]
    for _ in range(500):
        base = random.choice(points)
        i_val = base[0] + random.gauss(0, 0.1)
        q_val = base[1] + random.gauss(0, 0.1)
        constellation.append({"i": i_val, "q": q_val})
    
    return {
        "spectrum": spectrum,
        "waterfall": waterfall,
        "constellation": constellation
    }


@router.get("/{signal_id}/mock-results")
async def get_mock_results(signal_id: str):
    """
    Return mock analysis results matching frontend expectations.
    Temporary endpoint for MVP frontend development.
    """
    return {
        "parameters": {
            "carrierFrequency": 915.0e6,
            "sampleRate": 2.4e6,
            "bandwidth": 500000.0,
            "snr": 18.5,
            "symbolRate": 125000.0
        },
        "hypotheses": [
            {
                "id": "hyp-1",
                "modulation": "QPSK",
                "symbolRate": 125000.0,
                "mlConfidence": 0.89,
                "validation": {
                    "syncPassed": True,
                    "demodPassed": True,
                    "fecPassed": True
                },
                "isWinner": True
            },
            {
                "id": "hyp-2",
                "modulation": "BPSK",
                "symbolRate": 125000.0,
                "mlConfidence": 0.65,
                "validation": {
                    "syncPassed": True,
                    "demodPassed": True,
                    "fecPassed": False
                },
                "isWinner": False
            },
            {
                "id": "hyp-3",
                "modulation": "8PSK",
                "symbolRate": 125000.0,
                "mlConfidence": 0.42,
                "validation": {
                    "syncPassed": True,
                    "demodPassed": False,
                    "fecPassed": False
                },
                "isWinner": False
            },
            {
                "id": "hyp-4",
                "modulation": "16QAM",
                "symbolRate": 125000.0,
                "mlConfidence": 0.28,
                "validation": {
                    "syncPassed": False,
                    "demodPassed": False,
                    "fecPassed": False
                },
                "isWinner": False
            }
        ]
    }
