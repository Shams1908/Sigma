"""
Reporting helpers — assemble API response objects from stored Analysis data.

Converts db.models.Analysis documents (or in-memory equivalents) into the
Pydantic response schemas that match frontend/src/types/index.ts exactly.
"""
from __future__ import annotations

from typing import Any, Optional

from api.schemas import (
    EstimatedParameters,
    HypothesisCandidate,
    ReportResponse,
    ResultsResponse,
)


def build_results_response(
    analysis_id: str,
    signal_id: str,
    status: str,
    parameters: Optional[Any],
    hypotheses: list,
) -> ResultsResponse:
    """
    Convert raw Analysis data into the ResultsResponse schema.

    `parameters` may be a ParameterEstimate Beanie model or a plain dict.
    `hypotheses` may be Hypothesis Beanie models or plain dicts.
    """
    est_params = _build_estimated_parameters(parameters)
    candidates = [_build_hypothesis_candidate(h, i) for i, h in enumerate(hypotheses)]

    return ResultsResponse(
        analysis_id=analysis_id,
        signal_id=signal_id,
        status=status,
        parameters=est_params,
        hypotheses=candidates,
    )


def build_report_response(
    analysis_id: str,
    filename: str,
    status: str,
    parameters: Optional[Any],
    hypotheses: list,
    completed_at: Any = None,
) -> ReportResponse:
    """
    Build the exportable JSON summary for GET /results/{id}/report.
    Includes only the top hypothesis (rank 1) plus the full evidence list.
    """
    est_params = _build_estimated_parameters(parameters)
    candidates = [_build_hypothesis_candidate(h, i) for i, h in enumerate(hypotheses)]

    top = candidates[0] if candidates else None

    return ReportResponse(
        analysis_id=analysis_id,
        filename=filename,
        parameters=est_params,
        top_hypothesis=top,
        evidence=candidates,
        status=status,
        completed_at=completed_at,
    )


# ── Conversion helpers ─────────────────────────────────────────────────────────

def _build_estimated_parameters(parameters: Optional[Any]) -> Optional[EstimatedParameters]:
    """Accept either a Beanie ParameterEstimate model or a dict."""
    if parameters is None:
        return None

    if isinstance(parameters, dict):
        return EstimatedParameters(
            snr=float(parameters.get("snr", 0.0)),
            bandwidth=float(parameters.get("bandwidth", 0.0)),
            carrierOffset=float(parameters.get("carrier_offset", 0.0)),
            symbolRate=float(parameters.get("symbol_rate_estimate", 0.0)),
        )

    # Beanie model — access as attributes
    return EstimatedParameters(
        snr=float(getattr(parameters, "snr", 0.0)),
        bandwidth=float(getattr(parameters, "bandwidth", 0.0)),
        carrierOffset=float(getattr(parameters, "carrier_offset", 0.0)),
        symbolRate=float(getattr(parameters, "symbol_rate_estimate", 0.0)),
    )


def _build_hypothesis_candidate(h: Any, index: int) -> HypothesisCandidate:
    """
    Convert one Hypothesis doc / dict into a HypothesisCandidate schema.

    id field: use rank if available, else the list index + 1.
    status:
      "success"  → sync_pass AND demod_pass
      "failed"   → neither
      "pending"  → only sync_pass (demod not yet confirmed)
    """
    if isinstance(h, dict):
        sync_pass   = bool(h.get("sync_pass", False))
        demod_pass  = bool(h.get("demod_pass", False))
        modulation  = str(h.get("modulation", "UNKNOWN"))
        symbol_rate = float(h.get("symbol_rate", 0.0))
        score       = float(h.get("final_score", h.get("ml_confidence", 0.0)))
        rank        = int(h.get("rank", index + 1))
        fec_type    = h.get("fec_type")
    else:
        sync_pass   = bool(getattr(h, "sync_pass",  False))
        demod_pass  = bool(getattr(h, "demod_pass", False))
        modulation  = str(getattr(h, "modulation",  "UNKNOWN"))
        symbol_rate = float(getattr(h, "symbol_rate", 0.0))
        score       = float(getattr(h, "final_score",
                            getattr(h, "ml_confidence", 0.0)))
        rank        = int(getattr(h, "rank", index + 1))
        fec_type    = getattr(h, "fec_type", None)

    if sync_pass and demod_pass:
        status = "success"
    elif not sync_pass and not demod_pass:
        status = "failed"
    else:
        status = "pending"

    fec_label = f", FEC={fec_type}" if fec_type else ""
    details = (
        f"Rank {rank}: {modulation}{fec_label} @ "
        f"{symbol_rate:.0f} sym/s, score={score:.3f}"
    )

    return HypothesisCandidate(
        id=str(rank),
        modulation=modulation,
        symbolRate=symbol_rate,
        confidenceScore=score,
        details=details,
        status=status,
    )
