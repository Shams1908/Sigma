"""
Pydantic response schemas shared across API routers.
Field names mirror frontend/src/types/index.ts exactly where concepts overlap.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ── Upload ────────────────────────────────────────────────────────────────────

class SignalMetadata(BaseModel):
    """Matches frontend SignalMetadata interface exactly."""

    fileName: str
    sampleRate: float
    centerFrequency: float  # carrier offset used as proxy; 0.0 when unknown
    duration: float         # seconds; 0.0 when sample_rate not available
    fileSize: int           # bytes
    ingestionTime: str      # ISO-8601 datetime string


class UploadResponse(BaseModel):
    signal_id: str
    metadata: SignalMetadata


# ── Parameters ────────────────────────────────────────────────────────────────

class EstimatedParameters(BaseModel):
    """Matches frontend EstimatedParameters interface exactly."""

    snr: float
    bandwidth: float
    carrierOffset: float
    symbolRate: float


# ── Hypothesis ────────────────────────────────────────────────────────────────

class HypothesisCandidate(BaseModel):
    """Matches frontend HypothesisCandidate interface exactly."""

    id: str
    modulation: str
    symbolRate: float
    confidenceScore: float
    details: str
    status: Literal["pending", "success", "failed"]


# ── Analysis status ──────────────────────────────────────────────────────────

class AnalysisStatusResponse(BaseModel):
    analysis_id: str
    signal_id: str
    status: Literal["pending", "running", "done", "failed"]
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class AnalysisCreateResponse(BaseModel):
    analysis_id: str
    status: str


# ── Results ───────────────────────────────────────────────────────────────────

class ResultsResponse(BaseModel):
    analysis_id: str
    signal_id: str
    status: str
    parameters: Optional[EstimatedParameters] = None
    hypotheses: List[HypothesisCandidate] = []


class ReportResponse(BaseModel):
    """Exportable JSON summary of one analysis."""

    analysis_id: str
    filename: str
    parameters: Optional[EstimatedParameters] = None
    top_hypothesis: Optional[HypothesisCandidate] = None
    evidence: List[HypothesisCandidate] = []
    status: str
    completed_at: Optional[datetime] = None
