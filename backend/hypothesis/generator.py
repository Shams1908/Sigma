"""
Candidate hypothesis generator for SIGMA.
Generates candidate signal hypotheses from ML classifications, DSP parameter estimations,
and coding/interleaving configurations.
"""
import uuid
from typing import List, Dict, Optional, Any
try:
    from backend.hypothesis.candidates import (
        HypothesisCandidate,
        EvidenceTrace,
        EvidenceStatus,
        EvidenceComponent,
    )
    from backend.hypothesis.evaluator import create_ml_evidence
except ImportError:
    from hypothesis.candidates import (
        HypothesisCandidate,
        EvidenceTrace,
        EvidenceStatus,
        EvidenceComponent,
    )
    from hypothesis.evaluator import create_ml_evidence


def create_candidate(
    modulation: str,
    symbol_rate: float,
    fec_config: Optional[str] = None,
    interleaver_config: Optional[str] = None,
    sync_assumptions: Optional[Dict[str, Any]] = None,
    evidence: Optional[EvidenceTrace] = None,
    candidate_id: Optional[str] = None,
    details: str = "",
) -> HypothesisCandidate:
    """
    Constructs a well-formed HypothesisCandidate instance.
    """
    if candidate_id is None:
        clean_mod = modulation.replace("-", "").replace("/", "").lower()
        candidate_id = f"hyp_{clean_mod}_{int(symbol_rate)}_{uuid.uuid4().hex[:6]}"

    if evidence is None:
        evidence = EvidenceTrace()

    return HypothesisCandidate(
        id=candidate_id,
        modulation=modulation,
        symbolRate=float(symbol_rate),
        fec_config=fec_config,
        interleaver_config=interleaver_config,
        sync_assumptions=sync_assumptions,
        evidence=evidence,
        details=details,
        status="pending",
    )


def generate_hypotheses_from_ml(
    ml_probabilities: Dict[str, float],
    symbol_rate: float,
    supported_classes: Optional[List[str]] = None,
    fec_candidates: Optional[List[str]] = None,
    top_k: int = 5,
) -> List[HypothesisCandidate]:
    """
    Generates competing hypothesis candidates from ML classification probabilities and
    estimated symbol rate.
    
    Uses the genuine model prediction probabilities directly as ML evidence.
    Does NOT replace probabilities with hardcoded constants or fabricate scores
    for unsupported modulations.
    
    Args:
        ml_probabilities: Dictionary mapping modulation name to probability.
        symbol_rate: Estimated symbol rate in Baud.
        supported_classes: List of classes supported by the ML model.
        fec_candidates: Optional list of FEC configurations to test (e.g. ['none', 'conv_r1/2_k7']).
        top_k: Maximum number of top ML classes to generate hypotheses for.
        
    Returns:
        List[HypothesisCandidate]: Generated candidates ready for evaluation and ranking.
    """
    if not ml_probabilities:
        return []

    # Sort modulations by probability descending
    sorted_mods = sorted(ml_probabilities.items(), key=lambda item: item[1], reverse=True)[:top_k]

    candidates: List[HypothesisCandidate] = []
    fec_list = fec_candidates if fec_candidates else [None]

    for mod, prob in sorted_mods:
        ml_comp = create_ml_evidence(
            candidate_modulation=mod,
            ml_probabilities=ml_probabilities,
            supported_classes=supported_classes,
        )

        for fec in fec_list:
            fec_str = f" + {fec}" if fec and fec.lower() != "none" else ""
            evidence = EvidenceTrace(
                ml=ml_comp,
                constellation=EvidenceComponent(status=EvidenceStatus.NOT_EVALUATED),
                timing=EvidenceComponent(status=EvidenceStatus.NOT_EVALUATED),
                fec=EvidenceComponent(status=EvidenceStatus.NOT_EVALUATED),
                bitstream=EvidenceComponent(status=EvidenceStatus.NOT_EVALUATED),
            )

            cand = create_candidate(
                modulation=mod,
                symbol_rate=symbol_rate,
                fec_config=fec,
                evidence=evidence,
                details=f"Hypothesis: {mod}{fec_str} at {symbol_rate:g} Baud (ML prob: {prob:.4f})",
            )
            candidates.append(cand)

    return candidates
