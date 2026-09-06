"""
GET /api/v1/results/{analysis_id}          — full parameters + ranked hypotheses
GET /api/v1/results/{analysis_id}/report   — exportable JSON summary
"""
from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException

from api.schemas import ReportResponse, ResultsResponse
from db.init import is_db_connected

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Shared lookup helper ───────────────────────────────────────────────────────

async def _load_analysis(analysis_id: str) -> dict:
    """
    Load an Analysis record from DB or in-memory store.
    Returns a plain dict with keys:
        signal_id, status, parameters, hypotheses,
        error_message, completed_at, filename
    Raises HTTPException 404 if not found.
    """
    # Try DB
    if is_db_connected():
        try:
            from beanie import PydanticObjectId  # noqa: PLC0415
            from db.models import Analysis, Signal  # noqa: PLC0415

            doc = await Analysis.get(PydanticObjectId(analysis_id))
            if doc:
                filename = ""
                try:
                    sig = await Signal.get(doc.signal_id)
                    filename = sig.filename if sig else ""
                except Exception:  # noqa: BLE001
                    pass

                return {
                    "signal_id":    str(doc.signal_id),
                    "status":       doc.status,
                    "parameters":   doc.parameters,
                    "hypotheses":   doc.hypotheses,
                    "error_message": doc.error_message,
                    "completed_at": doc.completed_at,
                    "filename":     filename,
                }
        except Exception as exc:  # noqa: BLE001
            logger.warning("DB lookup failed for %s: %s", analysis_id, exc)

    # Fall back to in-memory store registered by analysis.py
    from api.analysis import _ANALYSIS_STORE  # noqa: PLC0415

    entry = _ANALYSIS_STORE.get(analysis_id)
    if not entry:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis '{analysis_id}' not found.",
        )

    # Resolve filename from upload store
    filename = ""
    try:
        from api.upload import _UPLOAD_STORE  # noqa: PLC0415
        storage_path = entry.get("storage_path") or _UPLOAD_STORE.get(
            entry.get("signal_id", ""), ""
        )
        if storage_path:
            from pathlib import Path
            filename = Path(storage_path).name
    except Exception:  # noqa: BLE001
        pass

    return {
        "signal_id":    entry.get("signal_id", ""),
        "status":       entry.get("status", "pending"),
        "parameters":   entry.get("parameters"),
        "hypotheses":   entry.get("hypotheses", []),
        "error_message": entry.get("error_message"),
        "completed_at": entry.get("completed_at"),
        "filename":     filename,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/{analysis_id}", response_model=ResultsResponse)
async def get_results(analysis_id: str):
    """
    Return the full parameter estimates and ranked hypothesis list for
    a completed analysis.

    Returns the data regardless of status — the client can inspect
    partial results while the analysis is still running.
    """
    data = await _load_analysis(analysis_id)

    from reporting.results import build_results_response  # noqa: PLC0415

    return build_results_response(
        analysis_id=analysis_id,
        signal_id=data["signal_id"],
        status=data["status"],
        parameters=data["parameters"],
        hypotheses=data["hypotheses"],
    )


@router.get("/{analysis_id}/report", response_model=ReportResponse)
async def get_report(analysis_id: str):
    """
    Return an exportable JSON summary: filename, parameters, top hypothesis,
    and full evidence list.  Designed for download / sharing.
    """
    data = await _load_analysis(analysis_id)

    if data["status"] not in ("done", "failed"):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Analysis is still '{data['status']}'. "
                "Report is only available after analysis completes."
            ),
        )

    from reporting.results import build_report_response  # noqa: PLC0415

    return build_report_response(
        analysis_id=analysis_id,
        filename=data["filename"],
        status=data["status"],
        parameters=data["parameters"],
        hypotheses=data["hypotheses"],
        completed_at=data["completed_at"],
    )


@router.get("/{analysis_id}/diagnostics")
async def get_diagnostics(analysis_id: str):
    """Get real EVM, timing, sync status from completed hypothesis evaluation."""
    data = await _load_analysis(analysis_id)
    
    hypotheses = data.get("hypotheses", [])
    params = data.get("parameters")
    
    if not hypotheses:
        return {
            "evm_rms": None,
            "timing_error_rms": None,
            "sync_locked": False,
            "demod_locked": False,
            "fec_valid": False,
            "snr": params.snr if params else 0.0,
            "carrier_offset": params.carrier_offset if params else 0.0,
        }
    
    best = hypotheses[0]
    
    return {
        "evm_rms": float(getattr(best, "evm_rms", 1.0)),
        "timing_error_rms": float(getattr(best, "timing_error_rms", 1.0)),
        "sync_locked": bool(getattr(best, "sync_pass", False)),
        "demod_locked": bool(getattr(best, "demod_pass", False)),
        "fec_valid": bool(getattr(best, "fec_pass", False)),
        "snr": float(params.snr) if params else 0.0,
        "carrier_offset": float(params.carrier_offset) if params else 0.0,
    }


@router.get("/{analysis_id}/bitstream")
async def get_bitstream(analysis_id: str):
    """Get real decoded bits from best hypothesis if available."""
    data = await _load_analysis(analysis_id)
    
    hypotheses = data.get("hypotheses", [])
    
    if not hypotheses:
        raise HTTPException(404, "No hypotheses available for bitstream extraction")
    
    best = hypotheses[0]
    
    decoded_bits = getattr(best, "decoded_bits", None)
    if decoded_bits is None:
        return {
            "available": False,
            "reason": "Bitstream not yet decoded or demodulation failed"
        }
    
    return {
        "available": True,
        "bits": decoded_bits[:1000] if len(decoded_bits) > 1000 else decoded_bits,
        "total_bits": len(decoded_bits),
        "entropy": float(getattr(best, "bit_entropy", 0.0)),
        "ones_ratio": float(getattr(best, "ones_ratio", 0.5)),
    }
