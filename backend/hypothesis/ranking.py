"""
Hypothesis deduplication, scoring, ranking, and normalized confidence computation.
Implements numerically stable softmax with configurable temperature and weight renormalization.
"""
import math
from typing import List, Optional, Dict, Any
try:
    from backend.core.config import settings, HypothesisScoringSettings
    from backend.hypothesis.candidates import HypothesisCandidate
    from backend.hypothesis.evaluator import calculate_raw_score
except ImportError:
    from core.config import settings, HypothesisScoringSettings
    from hypothesis.candidates import HypothesisCandidate
    from hypothesis.evaluator import calculate_raw_score


def deduplicate_candidates(candidates: List[HypothesisCandidate]) -> List[HypothesisCandidate]:
    """
    Deduplicates genuinely identical candidate configurations before confidence normalization.
    
    Candidates with distinct parameters (e.g. modulation, symbolRate, fec_config,
    interleaver_config, sync_assumptions) remain independent hypotheses.
    
    When identical candidates are found, the candidate with the higher rawScore
    (or first encountered as tie-breaker) is retained.
    
    Args:
        candidates: List of candidate hypotheses.
        
    Returns:
        List[HypothesisCandidate]: Deduplicated list preserving distinct hypotheses.
    """
    if not candidates:
        return []

    unique_candidates: Dict[Any, HypothesisCandidate] = {}

    for c in candidates:
        key = c.identity_key()
        if key not in unique_candidates:
            unique_candidates[key] = c
        else:
            # If duplicate configuration exists, retain the one with higher rawScore
            existing = unique_candidates[key]
            if c.rawScore > existing.rawScore:
                unique_candidates[key] = c

    return list(unique_candidates.values())


def normalize_confidences(
    candidates: List[HypothesisCandidate],
    temperature: Optional[float] = None,
    tolerance: Optional[float] = None,
) -> List[HypothesisCandidate]:
    """
    Jointly normalizes raw evidence scores across candidate hypotheses using
    numerically stable softmax.
    
    Formula:
        confidence_i = exp((raw_score_i - max_raw_score) / temperature) /
                       sum_j exp((raw_score_j - max_raw_score) / temperature)
                       
    Properties:
      1. Finite values in [0.0, 1.0].
      2. Sum of confidences across candidates approximately equals 1.0.
      3. Preserves raw score ranking order.
      4. Single candidate receives confidence 1.0.
      5. Equal raw scores receive equal confidences.
      
    Args:
        candidates: List of candidate hypotheses with computed rawScore.
        temperature: Softmax temperature parameter (T > 0). If None, uses settings.
        tolerance: Numerical tolerance for sum validation. If None, uses settings.
        
    Returns:
        List[HypothesisCandidate]: Candidates with updated confidenceScore fields.
    """
    if not candidates:
        return []

    if len(candidates) == 1:
        candidates[0].confidenceScore = 1.0
        return candidates

    if temperature is None:
        temperature = settings.HYPOTHESIS_SCORING.temperature
    if tolerance is None:
        tolerance = settings.HYPOTHESIS_SCORING.confidence_tolerance

    if temperature <= 0.0:
        raise ValueError(f"Softmax temperature must be strictly positive, got {temperature}")

    # Numerically stable softmax: subtract max_raw_score to prevent exp overflow
    max_raw = max(c.rawScore for c in candidates)
    scaled_diffs = [(c.rawScore - max_raw) / temperature for c in candidates]

    # Math exp with underflow clamp (-700 avoids math.exp underflow to 0 with exception in edge systems)
    exp_scores = [math.exp(max(-700.0, s)) for s in scaled_diffs]
    sum_exp = sum(exp_scores)

    if sum_exp <= 0.0:
        # Graceful fallback in extreme numerical cases
        uniform = 1.0 / len(candidates)
        for c in candidates:
            c.confidenceScore = uniform
    else:
        for c, exp_s in zip(candidates, exp_scores):
            c.confidenceScore = exp_s / sum_exp

    # Numerical validation of confidence properties
    conf_sum = sum(c.confidenceScore for c in candidates)
    if abs(conf_sum - 1.0) > tolerance:
        # Re-normalize if tiny precision drift occurs
        for c in candidates:
            c.confidenceScore = c.confidenceScore / conf_sum

    return candidates


def rank_hypotheses(
    candidates: List[HypothesisCandidate],
    config: Optional[HypothesisScoringSettings] = None,
    deduplicate: bool = True,
) -> List[HypothesisCandidate]:
    """
    Complete hypothesis scoring, deduplication, confidence normalization, and ranking workflow.
    
    Steps:
      1. Calculate raw evidence score for each candidate with available weight renormalization.
      2. Deduplicate genuinely identical candidate configurations (if enabled).
      3. Normalize confidences jointly across candidates using temperature-scaled softmax.
      4. Sort candidates descending by confidenceScore (and rawScore as tie-breaker).
      
    Args:
        candidates: List of candidate hypotheses to rank.
        config: Scoring configuration (weights, temperature). If None, uses settings.
        deduplicate: Whether to deduplicate identical configurations before normalization.
        
    Returns:
        List[HypothesisCandidate]: Ranked, deduplicated candidates with valid confidences.
    """
    if not candidates:
        return []

    if config is None:
        config = settings.HYPOTHESIS_SCORING

    # Step 1: Calculate raw scores for all candidates
    for c in candidates:
        calculate_raw_score(c, config)

    # Step 2: Deduplicate identical candidate configurations
    working_set = deduplicate_candidates(candidates) if deduplicate else list(candidates)

    # Step 3: Joint softmax confidence normalization
    working_set = normalize_confidences(
        working_set,
        temperature=config.temperature,
        tolerance=config.confidence_tolerance,
    )

    # Step 4: Sort descending by confidenceScore, then rawScore
    ranked = sorted(
        working_set,
        key=lambda c: (c.confidenceScore, c.rawScore),
        reverse=True,
    )

    # Step 5: Mark status and update details summary if not set
    for rank_idx, c in enumerate(ranked):
        if not c.details:
            c.details = (
                f"Rank #{rank_idx + 1}: {c.modulation} at {c.symbolRate:g} Baud. "
                f"Raw score: {c.rawScore:.4f}, Relative confidence: {c.confidenceScore * 100:.2f}%"
            )
        if c.status == "pending":
            c.status = "success" if rank_idx == 0 else "pending"

    return ranked
