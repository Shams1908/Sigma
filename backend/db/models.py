"""
Beanie ODM document definitions for the SIGMA platform.
All documents are stored in MongoDB Atlas via Motor (async driver).
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from beanie import Document, PydanticObjectId
from pydantic import BaseModel, Field


class ParameterEstimate(BaseModel):
    """DSP-derived parameter estimates for an ingested signal."""

    snr: float = Field(..., description="Estimated signal-to-noise ratio (dB)")
    carrier_offset: float = Field(
        ..., description="Estimated carrier frequency offset (Hz)"
    )
    bandwidth: float = Field(
        ..., description="Estimated occupied bandwidth (Hz)"
    )
    symbol_rate_estimate: float = Field(
        ..., description="Estimated symbol rate (sym/s)"
    )


class Hypothesis(BaseModel):
    """One candidate modulation hypothesis with per-stage pass/fail evidence."""

    modulation: str
    symbol_rate: float
    fec_type: Optional[str] = None

    # Confidence values
    ml_confidence: float
    calibrated_confidence: Optional[float] = None

    # Stage pass/fail flags
    sync_pass: bool
    demod_pass: bool
    fec_pass: bool
    bitstream_pass: bool

    # Composite score + rank
    final_score: float
    rank: int


class Signal(Document):
    """One uploaded signal file."""

    filename: str
    detected_format: str
    sample_rate: Optional[float] = None
    storage_path: str
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "signals"


class Analysis(Document):
    """Analysis job linked to a Signal document."""

    signal_id: PydanticObjectId
    status: Literal["pending", "running", "done", "failed"] = "pending"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    parameters: Optional[ParameterEstimate] = None
    hypotheses: List[Hypothesis] = []
    error_message: Optional[str] = None

    class Settings:
        name = "analyses"
