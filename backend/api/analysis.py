"""
API routes to trigger, parameterize, and monitor signal analysis pipelines (DSP, ML, Decoder Lab).
"""
from typing import List, Optional, Dict, Any
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Body, status
from pydantic import BaseModel, Field

try:
    from backend.hypothesis.decoder_contracts import DecoderConfig
    from backend.hypothesis.generator import HypothesisSearchConfig
    from backend.hypothesis.decoder_lab import run_decoder_lab, DecoderLabResult
    from backend.fec import encode_convolutional_r12_k7
    from backend.validation import append_crc_to_bits
except ImportError:
    from hypothesis.decoder_contracts import DecoderConfig
    from hypothesis.generator import HypothesisSearchConfig
    from hypothesis.decoder_lab import run_decoder_lab, DecoderLabResult
    from fec import encode_convolutional_r12_k7
    from validation import append_crc_to_bits

router = APIRouter()


# =====================================================================
# Request / Response Schemas
# =====================================================================

class DecoderLabApiRequest(BaseModel):
    """
    Request body for executing the Decoder Lab workflow.
    Accepts canonical IQ as either:
      - 'iq': 2D list of shape [2, N] (row 0 = I, row 1 = Q)
      - 'i_samples' & 'q_samples': separate equal-length 1D lists
    """
    iq: Optional[List[List[float]]] = Field(
        default=None,
        description="2D list of shape [2, N] where row 0 is I and row 1 is Q",
    )
    i_samples: Optional[List[float]] = Field(
        default=None,
        description="1D list of In-phase (I) samples",
    )
    q_samples: Optional[List[float]] = Field(
        default=None,
        description="1D list of Quadrature (Q) samples",
    )
    sample_rate: float = Field(
        ...,
        gt=0.0,
        description="Signal sampling rate in Hz",
        examples=[10000.0, 1000000.0],
    )
    symbol_rate: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Optional center symbol-rate estimate in Baud",
    )
    symbol_rate_uncertainty: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Optional symbol-rate uncertainty in Baud",
    )
    ml_probabilities: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional mapping from modulation name to probability (e.g. {'BPSK': 0.8, 'QPSK': 0.2})",
    )
    top_k_modulations: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Optional limit on number of top ML modulations to consider",
    )
    fec_candidates: Optional[List[str]] = Field(
        default=None,
        description="Optional list of FEC schemes to evaluate (e.g. ['none', 'conv_r1/2_k7'])",
    )
    crc_candidates: Optional[List[str]] = Field(
        default=None,
        description="Optional list of CRC schemes to evaluate (e.g. ['none', 'CRC-16-CCITT'])",
    )
    max_candidates: Optional[int] = Field(
        default=24,
        ge=1,
        le=24,
        description="Hard upper bound on generated candidates (maximum 24)",
    )
    enable_sync: bool = Field(default=True, description="Enable or bypass synchronization stage")
    enable_demod: bool = Field(default=True, description="Enable or bypass demodulation stage")
    enable_deinterleaver: bool = Field(default=True, description="Enable or bypass deinterleaving stage")
    enable_fec: bool = Field(default=True, description="Enable or bypass FEC stage")
    enable_validation: bool = Field(default=True, description="Enable or bypass validation stage")


class ValidatedDecodeSummary(BaseModel):
    id: str
    modulation: str
    symbol_rate: float
    confidence: float
    fec_config: Optional[str] = None
    interleaver_config: Optional[str] = None
    crc_scheme: Optional[str] = None
    bits_count: int = 0
    bits_preview: Optional[str] = None


class DecoderLabApiResponse(BaseModel):
    total_evaluated: int
    successful_decodes_count: int
    failed_decodes_count: int
    has_validated_decode: bool
    best_hypothesis_id: Optional[str] = None
    best_hypothesis_modulation: Optional[str] = None
    best_hypothesis_confidence: float = 0.0
    best_is_validated: bool = False
    validated_decode: Optional[ValidatedDecodeSummary] = None
    search_summary: Dict[str, Any] = Field(default_factory=dict)
    ranked_hypotheses: List[Dict[str, Any]] = Field(default_factory=list)
    decoder_results: Dict[str, Any] = Field(default_factory=dict)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


# =====================================================================
# Endpoints
# =====================================================================

@router.post(
    "/decoder-lab",
    response_model=DecoderLabApiResponse,
    summary="Execute Decoder Lab end-to-end hypothesis search and decoding",
    description=(
        "Accepts canonical IQ, performs bounded candidate hypothesis search (<= 24), "
        "executes the decoder chain per candidate with error isolation, aggregates multi-stage evidence, "
        "and returns ranked hypotheses, best hypothesis, and validated decode (if any)."
    ),
)
async def run_decoder_lab_endpoint(
    req: DecoderLabApiRequest = Body(...),
) -> DecoderLabApiResponse:
    # 1. Parse and validate IQ representation
    if req.iq is not None:
        if len(req.iq) != 2:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Field 'iq' must have shape [2, N] (2 channels: I and Q). Got {len(req.iq)} channels.",
            )
        i_data = req.iq[0]
        q_data = req.iq[1]
        if len(i_data) != len(q_data):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Channel length mismatch: I has {len(i_data)} samples, Q has {len(q_data)} samples.",
            )
        if len(i_data) == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="IQ arrays contain 0 samples.",
            )
        iq_array = np.vstack([np.asarray(i_data, dtype=np.float32), np.asarray(q_data, dtype=np.float32)])
    elif req.i_samples is not None and req.q_samples is not None:
        if len(req.i_samples) != len(req.q_samples):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Mismatched sample lengths: i_samples has {len(req.i_samples)}, q_samples has {len(req.q_samples)}.",
            )
        if len(req.i_samples) == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Sample arrays contain 0 samples.",
            )
        iq_array = np.vstack([np.asarray(req.i_samples, dtype=np.float32), np.asarray(req.q_samples, dtype=np.float32)])
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Missing IQ signal: provide either 'iq' ([[I], [Q]]) or both 'i_samples' and 'q_samples'.",
        )

    # 2. Build search configuration
    search_kwargs: Dict[str, Any] = {"max_candidates": min(24, req.max_candidates or 24)}
    if req.top_k_modulations is not None:
        search_kwargs["top_k_modulations"] = req.top_k_modulations
    if req.fec_candidates is not None:
        search_kwargs["fec_candidates"] = req.fec_candidates
    if req.crc_candidates is not None:
        search_kwargs["crc_candidates"] = req.crc_candidates

    search_cfg = HypothesisSearchConfig(**search_kwargs)

    # 3. Build decoder stage configuration
    decoder_cfg = DecoderConfig(
        enable_sync=req.enable_sync,
        enable_demod=req.enable_demod,
        enable_deinterleaver=req.enable_deinterleaver,
        enable_fec=req.enable_fec,
        enable_validation=req.enable_validation,
    )

    # 4. Execute DecoderLab
    try:
        lab_res: DecoderLabResult = run_decoder_lab(
            iq=iq_array,
            sample_rate=req.sample_rate,
            symbol_rate=req.symbol_rate,
            symbol_rate_uncertainty=req.symbol_rate_uncertainty,
            ml_probabilities=req.ml_probabilities,
            search_config=search_cfg,
            decoder_config=decoder_cfg,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Decoder Lab validation error: {str(val_err)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Decoder Lab internal execution error: {str(exc)}",
        )

    # 5. Return sanitized dictionary response (Pydantic model validates to_dict)
    return DecoderLabApiResponse(**lab_res.to_dict())


@router.get(
    "/decoder-lab/demo",
    response_model=DecoderLabApiResponse,
    summary="Run deterministic Decoder Lab demonstration with synthetic BPSK+FEC+CRC",
    description=(
        "Generates a deterministic synthetic BPSK frame with convolutional coding (Rate 1/2, K=7) "
        "and CRC-16-CCITT checksum, runs the Decoder Lab workflow, and returns the full result."
    ),
)
async def run_decoder_lab_demo_endpoint() -> DecoderLabApiResponse:
    """
    Deterministic demonstration endpoint using fixed synthetic BPSK + Rate 1/2 K=7 + CRC-16-CCITT.
    """
    # Deterministic payload (16 known bits)
    payload = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)

    # 1. Append CRC-16-CCITT
    payload_with_crc = append_crc_to_bits(payload, crc_name="CRC-16-CCITT")

    # 2. Convolutional encode (Rate 1/2, K=7)
    encoded_bits = encode_convolutional_r12_k7(payload_with_crc)

    # 3. BPSK modulation (1 sample per symbol, real channel)
    bpsk_symbols = np.where(encoded_bits == 1, 1.0, -1.0).astype(np.float32)
    iq_array = np.vstack([bpsk_symbols, np.zeros_like(bpsk_symbols)]).astype(np.float32)

    # 4. Search configuration
    search_cfg = HypothesisSearchConfig(
        top_k_modulations=2,
        fec_candidates=["none", "conv_r1/2_k7"],
        crc_candidates=["none", "CRC-16-CCITT"],
        max_candidates=10,
    )

    # 5. Run DecoderLab
    lab_res = run_decoder_lab(
        iq=iq_array,
        sample_rate=1000.0,
        symbol_rate=1000.0,
        ml_probabilities={"BPSK": 0.85, "QPSK": 0.15},
        search_config=search_cfg,
    )

    return DecoderLabApiResponse(**lab_res.to_dict())
