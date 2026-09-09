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
from backend.hypothesis.decoder_contracts import (
    DecoderStageStatus,
    SynchronizationResult,
    DemodulationResult,
    InterleaverResult,
    FECResult,
    ValidationResult,
    DecoderConfig,
    DecoderRequest,
    DecoderContext,
    DecoderResult,
    DecoderEngine,
    ScaffoldDecoderEngine,
    decoder_result_to_evidence_trace,
    update_hypothesis_from_decoder_result,
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
    "DecoderStageStatus",
    "SynchronizationResult",
    "DemodulationResult",
    "InterleaverResult",
    "FECResult",
    "ValidationResult",
    "DecoderConfig",
    "DecoderRequest",
    "DecoderContext",
    "DecoderResult",
    "DecoderEngine",
    "ScaffoldDecoderEngine",
    "decoder_result_to_evidence_trace",
    "update_hypothesis_from_decoder_result",
]

