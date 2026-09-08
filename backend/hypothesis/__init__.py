"""
Hypothesis package initialization.
Exports candidate models, evidence structures, scoring functions, and ranking utilities.
"""
from backend.hypothesis.candidates import (
    EvidenceStatus,
    EvidenceComponent,
    EvidenceTrace,
    HypothesisCandidate,
)
from backend.hypothesis.evaluator import (
    calculate_raw_score,
    create_ml_evidence,
    create_constellation_evidence,
    create_timing_evidence,
    create_fec_evidence,
    create_bitstream_evidence,
)
from backend.hypothesis.ranking import (
    deduplicate_candidates,
    normalize_confidences,
    rank_hypotheses,
)
from backend.hypothesis.generator import (
    create_candidate,
    generate_hypotheses_from_ml,
)

__all__ = [
    "EvidenceStatus",
    "EvidenceComponent",
    "EvidenceTrace",
    "HypothesisCandidate",
    "calculate_raw_score",
    "create_ml_evidence",
    "create_constellation_evidence",
    "create_timing_evidence",
    "create_fec_evidence",
    "create_bitstream_evidence",
    "deduplicate_candidates",
    "normalize_confidences",
    "rank_hypotheses",
    "create_candidate",
    "generate_hypotheses_from_ml",
]
