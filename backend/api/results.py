"""
API routes to retrieve analysis results, parameter estimation logs, and hypothesis rankings.
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field

try:
    from backend.hypothesis.candidates import (
        HypothesisCandidate,
        EvidenceTrace,
        EvidenceComponent,
        EvidenceStatus,
    )
    from backend.hypothesis.ranking import rank_hypotheses
    from backend.hypothesis.generator import generate_hypotheses_from_ml, create_candidate
    from backend.core.config import settings, HypothesisScoringSettings
except ImportError:
    from hypothesis.candidates import (
        HypothesisCandidate,
        EvidenceTrace,
        EvidenceComponent,
        EvidenceStatus,
    )
    from hypothesis.ranking import rank_hypotheses
    from hypothesis.generator import generate_hypotheses_from_ml, create_candidate
    from core.config import settings, HypothesisScoringSettings

router = APIRouter()


class EvidenceComponentSchema(BaseModel):
    status: str
    score: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class EvidenceTraceSchema(BaseModel):
    ml: Optional[EvidenceComponentSchema] = None
    mlModulation: Optional[EvidenceComponentSchema] = None
    symbolRate: Optional[EvidenceComponentSchema] = None
    symbol_rate: Optional[EvidenceComponentSchema] = None
    snr: Optional[EvidenceComponentSchema] = None
    constellation: Optional[EvidenceComponentSchema] = None
    timing: Optional[EvidenceComponentSchema] = None
    fec: Optional[EvidenceComponentSchema] = None
    bitstream: Optional[EvidenceComponentSchema] = None


class HypothesisCandidateSchema(BaseModel):
    id: str
    modulation: str
    symbolRate: float
    rawScore: Optional[float] = 0.0
    confidenceScore: Optional[float] = 0.0
    details: Optional[str] = ""
    status: Optional[str] = "pending"
    fec_config: Optional[str] = None
    interleaver_config: Optional[str] = None
    sync_assumptions: Optional[Dict[str, Any]] = None
    evidence: Optional[Dict[str, Any]] = None


class RankHypothesesRequest(BaseModel):
    candidates: List[HypothesisCandidateSchema]
    temperature: Optional[float] = Field(default=None, description="Optional override for softmax temperature")
    weights: Optional[Dict[str, float]] = Field(default=None, description="Optional override for component weights")


class RankHypothesesResponse(BaseModel):
    candidates: List[Dict[str, Any]]
    total_candidates: int
    temperature_used: float


@router.get("/hypotheses", response_model=RankHypothesesResponse)
async def get_hypotheses_ranking(
    symbol_rate: float = Query(default=1000.0, description="Estimated symbol rate in Baud"),
    temperature: Optional[float] = Query(default=None, description="Softmax temperature override"),
):
    """
    Retrieves ranked signal decoding hypotheses.
    Demonstrates dynamic raw-score evidence weighting and temperature-scaled softmax normalization.
    """
    # Sample multi-candidate demonstration
    demo_probabilities = {
        "BPSK": 0.82,
        "QPSK": 0.12,
        "8PSK": 0.04,
        "QAM16": 0.015,
        "QAM64": 0.005,
    }
    
    candidates = generate_hypotheses_from_ml(
        ml_probabilities=demo_probabilities,
        symbol_rate=symbol_rate,
        fec_candidates=["none", "conv_r1/2_k7"],
    )

    custom_config = None
    if isinstance(temperature, (int, float)):
        config_dict = settings.HYPOTHESIS_SCORING.model_dump()
        config_dict["temperature"] = float(temperature)
        custom_config = HypothesisScoringSettings(**config_dict)

    ranked = rank_hypotheses(candidates, config=custom_config)
    temp_used = custom_config.temperature if custom_config else settings.HYPOTHESIS_SCORING.temperature

    return RankHypothesesResponse(
        candidates=[c.to_dict() for c in ranked],
        total_candidates=len(ranked),
        temperature_used=temp_used,
    )


@router.post("/hypotheses/rank", response_model=RankHypothesesResponse)
async def rank_candidates_endpoint(
    request: RankHypothesesRequest = Body(...)
):
    """
    Accepts candidate hypotheses and computes:
      1. Dynamic raw evidence scores with weight renormalization.
      2. Deduplication of genuinely identical candidate configurations.
      3. Temperature-scaled softmax normalized relative confidences across candidates.
    """
    try:
        candidate_objs: List[HypothesisCandidate] = []
        for c_schema in request.candidates:
            # Reconstruct evidence trace if provided
            evidence = EvidenceTrace()
            if c_schema.evidence:
                dim_aliases = {
                    "ml": "ml",
                    "mlModulation": "ml",
                    "symbol_rate": "symbol_rate",
                    "symbolRate": "symbol_rate",
                    "snr": "snr",
                    "constellation": "constellation",
                    "timing": "timing",
                    "fec": "fec",
                    "bitstream": "bitstream",
                }
                for dim_key, target_attr in dim_aliases.items():
                    dim_data = c_schema.evidence.get(dim_key)
                    if isinstance(dim_data, dict):
                        status_str = dim_data.get("status", "not_evaluated")
                        try:
                            status_enum = EvidenceStatus(status_str)
                        except ValueError:
                            status_enum = EvidenceStatus.NOT_EVALUATED
                        score = dim_data.get("score")
                        details = dim_data.get("details")
                        comp = EvidenceComponent(status=status_enum, score=score, details=details)
                        setattr(evidence, target_attr, comp)

            cand = create_candidate(
                modulation=c_schema.modulation,
                symbol_rate=c_schema.symbolRate,
                fec_config=c_schema.fec_config,
                interleaver_config=c_schema.interleaver_config,
                sync_assumptions=c_schema.sync_assumptions,
                evidence=evidence,
                candidate_id=c_schema.id,
                details=c_schema.details or "",
            )
            candidate_objs.append(cand)

        # Build config override if requested
        config_dict = settings.HYPOTHESIS_SCORING.model_dump()
        if request.temperature is not None:
            config_dict["temperature"] = request.temperature
        if request.weights:
            for w_name, w_val in request.weights.items():
                if w_name in config_dict:
                    config_dict[w_name] = w_val

        custom_config = HypothesisScoringSettings(**config_dict)
        ranked = rank_hypotheses(candidate_objs, config=custom_config)

        return RankHypothesesResponse(
            candidates=[c.to_dict() for c in ranked],
            total_candidates=len(ranked),
            temperature_used=custom_config.temperature,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
