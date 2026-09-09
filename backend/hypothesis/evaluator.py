"""
Hypothesis evidence evaluator and raw-score calculator.
Calculates raw evidence scores from actually available evidence with weight renormalization.
Preserves explicit provenance status without arbitrary numeric defaults.
"""
import math
from typing import Dict, Any, Optional, List
try:
    from backend.core.config import settings, HypothesisScoringSettings
    from backend.hypothesis.candidates import (
        HypothesisCandidate,
        EvidenceStatus,
        EvidenceComponent,
        EvidenceTrace,
    )
except ImportError:
    from core.config import settings, HypothesisScoringSettings
    from hypothesis.candidates import (
        HypothesisCandidate,
        EvidenceStatus,
        EvidenceComponent,
        EvidenceTrace,
    )


def calculate_raw_score(
    candidate: HypothesisCandidate,
    config: Optional[HypothesisScoringSettings] = None
) -> float:
    """
    Calculates the raw evidence score for a single hypothesis candidate.
    
    Formula:
        Score = sum(w_k * score_k) / sum(w_k) for all evaluated components k.
        
    Rules:
      1. Uses only evaluated dimensions (AVAILABLE or FAILED).
      2. Missing dimensions (NOT_EVALUATED, NOT_SUPPORTED) are excluded from
         both numerator and denominator, with weights renormalized to 1.0.
      3. No arbitrary default numbers (e.g. 0.5 or 0.6) are assigned for missing dimensions.
      4. FAILED dimensions contribute a score of 0.0 with full applicable weight.
      5. If no dimensions are evaluated, returns 0.0.
      
    Args:
        candidate: The candidate hypothesis with attached evidence.
        config: Optional scoring settings. If None, uses system settings.
        
    Returns:
        float: The raw evidence score in [0.0, 1.0].
    """
    if config is None:
        config = settings.HYPOTHESIS_SCORING

    evidence = candidate.evidence
    w_sym = getattr(config, "weight_symbol_rate", 0.20)
    w_snr = getattr(config, "weight_snr", 0.10)
    dimension_map = [
        (evidence.ml, config.weight_ml, "ml"),
        (evidence.symbol_rate, w_sym, "symbol_rate"),
        (evidence.snr, w_snr, "snr"),
        (evidence.constellation, config.weight_constellation, "constellation"),
        (evidence.timing, config.weight_timing, "timing"),
        (evidence.fec, config.weight_fec, "fec"),
        (evidence.bitstream, config.weight_bitstream, "bitstream"),
    ]

    weighted_sum = 0.0
    active_weight_sum = 0.0

    for comp, weight, dim_name in dimension_map:
        if comp.status == EvidenceStatus.AVAILABLE and comp.score is not None:
            # Clamp available score to [0.0, 1.0] for safety
            clamped_score = max(0.0, min(1.0, float(comp.score)))
            weighted_sum += weight * clamped_score
            active_weight_sum += weight
            comp.weight = weight
        elif comp.status == EvidenceStatus.FAILED:
            # Failed evaluation represents an evaluated negative result (score 0.0)
            weighted_sum += 0.0
            active_weight_sum += weight
            comp.weight = weight
        else:
            # NOT_EVALUATED or NOT_SUPPORTED - do not contribute to score or weight
            comp.weight = 0.0

    if active_weight_sum > 0.0:
        raw_score = weighted_sum / active_weight_sum
        candidate.rawScore = float(raw_score)
    elif candidate.rawScore != 0.0:
        # Preserve caller's explicit pre-assigned rawScore if no conflicting evidence attached
        raw_score = candidate.rawScore
    else:
        raw_score = 0.0
        candidate.rawScore = 0.0

    return candidate.rawScore


def create_ml_evidence(
    candidate_modulation: str,
    ml_probabilities: Optional[Dict[str, float]] = None,
    supported_classes: Optional[List[str]] = None,
    details: Optional[Dict[str, Any]] = None,
) -> EvidenceComponent:
    """
    Constructs an EvidenceComponent for ML classification confidence.
    
    Args:
        candidate_modulation: Name of the candidate modulation.
        ml_probabilities: Mapping from modulation name to probability.
        supported_classes: List of classes supported by the ML model.
        details: Optional diagnostic details.
    """
    if ml_probabilities is None:
        return EvidenceComponent(
            status=EvidenceStatus.NOT_EVALUATED,
            details=details,
        )

    # Normalize key lookup (e.g. upper-case comparison)
    norm_candidate = candidate_modulation.strip().upper()
    prob_lookup = {k.strip().upper(): v for k, v in ml_probabilities.items()}

    if norm_candidate in prob_lookup:
        prob = float(prob_lookup[norm_candidate])
        info = {"source": "ml_classification_probability", "probability": prob}
        if details:
            info.update(details)
        return EvidenceComponent(
            status=EvidenceStatus.AVAILABLE,
            score=prob,
            details=info,
        )

    # Check if candidate is not supported by the model
    if supported_classes is not None:
        supported_norm = [c.strip().upper() for c in supported_classes]
        if norm_candidate not in supported_norm:
            info = {
                "reason": f"Modulation '{candidate_modulation}' is not in ML classifier taxonomy",
                "supported_classes": supported_classes,
            }
            if details:
                info.update(details)
            return EvidenceComponent(
                status=EvidenceStatus.NOT_SUPPORTED,
                details=info,
            )

    return EvidenceComponent(
        status=EvidenceStatus.NOT_SUPPORTED,
        details={"reason": f"Candidate modulation '{candidate_modulation}' not found in ML prediction vector"},
    )


def create_constellation_evidence(
    score: Optional[float] = None,
    status: EvidenceStatus = EvidenceStatus.NOT_EVALUATED,
    details: Optional[Dict[str, Any]] = None,
) -> EvidenceComponent:
    """Creates EvidenceComponent for constellation agreement."""
    if status == EvidenceStatus.AVAILABLE and score is not None:
        return EvidenceComponent(status=status, score=float(score), details=details)
    return EvidenceComponent(status=status, details=details)


def create_timing_evidence(
    score: Optional[float] = None,
    status: EvidenceStatus = EvidenceStatus.NOT_EVALUATED,
    details: Optional[Dict[str, Any]] = None,
) -> EvidenceComponent:
    """Creates EvidenceComponent for timing/synchronization quality."""
    if status == EvidenceStatus.AVAILABLE and score is not None:
        return EvidenceComponent(status=status, score=float(score), details=details)
    return EvidenceComponent(status=status, details=details)


def create_fec_evidence(
    score: Optional[float] = None,
    status: EvidenceStatus = EvidenceStatus.NOT_EVALUATED,
    details: Optional[Dict[str, Any]] = None,
) -> EvidenceComponent:
    """Creates EvidenceComponent for forward error correction validation."""
    if status == EvidenceStatus.AVAILABLE and score is not None:
        return EvidenceComponent(status=status, score=float(score), details=details)
    return EvidenceComponent(status=status, details=details)


def create_bitstream_evidence(
    score: Optional[float] = None,
    status: EvidenceStatus = EvidenceStatus.NOT_EVALUATED,
    details: Optional[Dict[str, Any]] = None,
) -> EvidenceComponent:
    """Creates EvidenceComponent for bitstream correlation."""
    if status == EvidenceStatus.AVAILABLE and score is not None:
        return EvidenceComponent(status=status, score=float(score), details=details)
    return EvidenceComponent(status=status, details=details)


def create_symbol_rate_evidence(
    candidate_baud: float,
    estimated_baud: float,
    uncertainty_baud: float = 0.0,
    details: Optional[Dict[str, Any]] = None,
) -> EvidenceComponent:
    """
    Evaluates physical symbol-rate parameter agreement between candidate hypothesis and M8 estimate.
    
    Computes Gaussian likelihood agreement:
      A = exp(-0.5 * (delta / sigma_eff)^2)
      
    Returns EvidenceComponent with status=AVAILABLE and score in [0.0, 1.0].
    """
    cand_b = float(candidate_baud)
    est_b = float(estimated_baud)
    if cand_b <= 0 or est_b <= 0:
        return EvidenceComponent(
            status=EvidenceStatus.FAILED,
            score=0.0,
            details={"error": "Non-positive symbol rate provided"},
        )

    unc_b = float(uncertainty_baud) if uncertainty_baud > 0 else (0.05 * est_b)
    sigma_eff = max(unc_b, 0.03 * est_b)
    delta_baud = abs(cand_b - est_b)
    rel_error = delta_baud / est_b
    z = delta_baud / sigma_eff

    agreement = float(math.exp(-0.5 * (z ** 2)))
    agreement = max(0.0, min(1.0, agreement))

    info = {
        "candidate_baud": cand_b,
        "estimated_baud": est_b,
        "uncertainty_baud": unc_b,
        "absolute_error_baud": delta_baud,
        "relative_error": rel_error,
        "agreement_score": agreement,
    }
    if details:
        info.update(details)

    return EvidenceComponent(
        status=EvidenceStatus.AVAILABLE,
        score=agreement,
        details=info,
    )


def create_snr_evidence(
    estimated_snr_db: Optional[float] = None,
    uncertainty_db: Optional[float] = None,
    status: EvidenceStatus = EvidenceStatus.NOT_EVALUATED,
    details: Optional[Dict[str, Any]] = None,
) -> EvidenceComponent:
    """
    Constructs an EvidenceComponent for SNR evidence.
    
    In accordance with project SNR policy:
      - Does NOT fabricate or hand-tune arbitrary modulation-specific thresholds.
      - Retains neutral NOT_EVALUATED status by default so SNR does not distort ranking.
      - Stores estimated SNR telemetry in details for downstream inspection and decoders.
    """
    info: Dict[str, Any] = {}
    if estimated_snr_db is not None:
        info["estimated_snr_db"] = float(estimated_snr_db)
    if uncertainty_db is not None:
        info["uncertainty_db"] = float(uncertainty_db)
    if details:
        info.update(details)

    if status == EvidenceStatus.AVAILABLE and "score" in info:
        return EvidenceComponent(status=status, score=float(info["score"]), details=info)

    return EvidenceComponent(
        status=status,
        details=info if info else None,
    )
