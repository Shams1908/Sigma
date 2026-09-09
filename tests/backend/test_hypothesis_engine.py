"""
Unit and integration tests for P5.5 Evidence Aggregation & Hypothesis Ranking Integration.

Verifies:
  1. Multi-candidate independent execution and decoder result mapping.
  2. Strict separation of semantics:
     - best_hypothesis = highest-confidence ranked candidate (regardless of decode success)
     - validated_decode = highest-ranked candidate where DecoderResult.success == True
  3. Scenario A: Validated candidate is also best_hypothesis.
  4. Scenario B: Best hypothesis is NOT validated (high ML/constellation confidence, failed decode;
     lower-confidence candidate succeeds and becomes validated_decode).
  5. Interleaver evidence is provenance-only (weight 0.0, zero score contribution).
  6. Decoder exceptions use canonical evidence path (canonical DecoderResult -> decoder_result_to_evidence_trace).
  7. Error isolation: An unexpected exception in one candidate does not abort evaluation of other candidates.
  8. Preservation of pre-existing ML, M8 symbol rate, and SNR telemetry.
  9. Missing and unsupported stages (NOT_EVALUATED, NOT_SUPPORTED) remain unscored with weights renormalized.
  10. Confidence normalization: Softmax ensures confidences in [0, 1] and sum to 1.0.
  11. Deduplication handling and deterministic repeated execution.
  12. Full synthetic end-to-end integration:
      P5.2 candidate generation -> P5.3 DecoderChain -> P5.4 genuine decoding/validation -> evidence mapping -> ranking.
"""
import math
import numpy as np
import pytest

from backend.core.config import settings, HypothesisScoringSettings
from backend.hypothesis.candidates import (
    HypothesisCandidate,
    EvidenceStatus,
    EvidenceComponent,
    EvidenceTrace,
)
from backend.hypothesis.evaluator import (
    calculate_raw_score,
    create_ml_evidence,
    create_symbol_rate_evidence,
    create_snr_evidence,
    create_constellation_evidence,
    create_fec_evidence,
    create_bitstream_evidence,
    create_interleaver_evidence,
)
from backend.hypothesis.ranking import rank_hypotheses
from backend.hypothesis.generator import create_candidate
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
    decoder_result_to_evidence_trace,
    update_hypothesis_from_decoder_result,
)
from backend.hypothesis.decoder_chain import (
    DecoderChain,
    SynchronizationAdapter,
    DemodulationAdapter,
    InterleaverAdapter,
    FECAdapter,
    ValidationAdapter,
)
from backend.hypothesis.engine import (
    HypothesisEngine,
    HypothesisEngineResult,
)
from backend.fec import encode_convolutional_r12_k7, ConvolutionalFECDecoder
from backend.interleaver import BlockDeinterleaver
from backend.validation import append_crc_to_bits, FrameValidator


# =====================================================================
# Test Helpers
# =====================================================================

def _make_synthetic_iq(num_samples: int = 200, freq: float = 0.05) -> np.ndarray:
    """Creates a canonical 2D real float32 array [2, N]."""
    t = np.arange(num_samples, dtype=np.float32)
    i = np.cos(2.0 * np.pi * freq * t).astype(np.float32)
    q = np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    return np.vstack([i, q])


def _make_ml_evidence(modulation: str, score: float) -> EvidenceComponent:
    """Helper to create an ML evidence component with given modulation probability."""
    return create_ml_evidence(candidate_modulation=modulation, ml_probabilities={modulation: score})


# =====================================================================
# 1. Basic Engine Execution & Lifecycle
# =====================================================================

def test_engine_empty_candidates():
    """Verifies that executing an empty candidate list returns an empty result cleanly."""
    engine = HypothesisEngine()
    iq = _make_synthetic_iq(100)
    result = engine.execute_and_rank(iq=iq, sample_rate=10000.0, candidates=[])

    assert result.total_evaluated == 0
    assert result.ranked_hypotheses == []
    assert result.best_hypothesis is None
    assert result.validated_decode is None
    assert result.successful_decodes_count == 0
    assert result.failed_decodes_count == 0


def test_multi_candidate_independent_execution():
    """Verifies that multiple candidates execute independently with per-candidate DecoderResults."""
    cand1 = create_candidate(modulation="BPSK", symbol_rate=1000.0)
    cand2 = create_candidate(modulation="QPSK", symbol_rate=2000.0)
    cand3 = create_candidate(modulation="16QAM", symbol_rate=1000.0)

    engine = HypothesisEngine()
    iq = _make_synthetic_iq(200)
    res = engine.execute_and_rank(
        iq=iq,
        sample_rate=10000.0,
        candidates=[cand1, cand2, cand3],
    )

    assert res.total_evaluated == 3
    assert len(res.decoder_results) == 3
    assert cand1.id in res.decoder_results
    assert cand2.id in res.decoder_results
    assert cand3.id in res.decoder_results

    # 16QAM demodulation is not implemented -> NOT_SUPPORTED
    assert res.decoder_results[cand3.id].demodulation.status == DecoderStageStatus.NOT_SUPPORTED
    assert res.decoder_results[cand3.id].success is False


# =====================================================================
# 2. Decoder Result to Evidence Mapping
# =====================================================================

def test_decoder_results_mapped_to_evidence_trace():
    """Verifies that stage results are mapped into candidate.evidence components."""
    cand = create_candidate(modulation="BPSK", symbol_rate=1000.0)
    engine = HypothesisEngine()
    iq = _make_synthetic_iq(200)

    res = engine.execute_and_rank(iq=iq, sample_rate=10000.0, candidates=[cand])
    evaluated_cand = res.ranked_hypotheses[0]

    # Timing and constellation should be AVAILABLE
    assert evaluated_cand.evidence.timing.status == EvidenceStatus.AVAILABLE
    assert evaluated_cand.evidence.timing.score is not None
    assert evaluated_cand.evidence.constellation.status == EvidenceStatus.AVAILABLE
    assert evaluated_cand.evidence.constellation.score is not None

    # FEC and interleaver unconfigured -> NOT_EVALUATED
    assert evaluated_cand.evidence.fec.status == EvidenceStatus.NOT_EVALUATED
    assert evaluated_cand.evidence.interleaver.status == EvidenceStatus.NOT_EVALUATED


def test_preservation_of_ml_and_m8_evidence():
    """
    Verifies that pre-existing ML, M8 symbol-rate, and SNR evidence attached to
    the candidate are preserved without corruption or erasure during decoding.
    """
    cand = create_candidate(modulation="BPSK", symbol_rate=1000.0)
    cand.evidence.ml = _make_ml_evidence("BPSK", 0.92)
    cand.evidence.symbol_rate = create_symbol_rate_evidence(candidate_baud=1000.0, estimated_baud=1005.0, uncertainty_baud=10.0)
    cand.evidence.snr = create_snr_evidence(estimated_snr_db=18.5, status=EvidenceStatus.NOT_EVALUATED)

    engine = HypothesisEngine()
    iq = _make_synthetic_iq(200)
    res = engine.execute_and_rank(iq=iq, sample_rate=10000.0, candidates=[cand])

    evaluated = res.ranked_hypotheses[0]
    assert evaluated.evidence.ml.status == EvidenceStatus.AVAILABLE
    assert evaluated.evidence.ml.score == pytest.approx(0.92, abs=1e-4)

    assert evaluated.evidence.symbol_rate.status == EvidenceStatus.AVAILABLE
    assert evaluated.evidence.symbol_rate.details["candidate_baud"] == 1000.0
    assert evaluated.evidence.symbol_rate.details["estimated_baud"] == 1005.0

    assert evaluated.evidence.snr.status == EvidenceStatus.NOT_EVALUATED
    assert evaluated.evidence.snr.details["estimated_snr_db"] == 18.5


def test_missing_unsupported_stages_unscored():
    """
    Verifies that NOT_SUPPORTED and NOT_EVALUATED stages have score=None and
    are excluded from raw score calculation, renormalizing over available dimensions only.
    """
    cand = create_candidate(modulation="BPSK", symbol_rate=1000.0)
    cand.evidence.ml = _make_ml_evidence("BPSK", 0.80)

    # Set mock decoder result with NOT_SUPPORTED validation and NOT_EVALUATED fec
    dec_res = DecoderResult(
        success=False,
        hypothesis_id=cand.id,
        synchronization=SynchronizationResult.create_success(timing_score=0.90),
        demodulation=DemodulationResult.create_success(constellation_score=0.85),
        interleaver=InterleaverResult.not_evaluated(),
        fec=FECResult.not_evaluated(),
        validation=ValidationResult.not_supported(failure_reason="No CRC scheme configured"),
    )
    update_hypothesis_from_decoder_result(cand, dec_res)

    assert cand.evidence.fec.status == EvidenceStatus.NOT_EVALUATED
    assert cand.evidence.fec.score is None
    assert cand.evidence.bitstream.status == EvidenceStatus.NOT_SUPPORTED
    assert cand.evidence.bitstream.score is None

    # Raw score must only use ml (0.80), timing (0.90), constellation (0.85)
    score = calculate_raw_score(cand)
    assert 0.80 <= score <= 0.90


# =====================================================================
# 3. Mandatory Requirement 2: Interleaver Provenance-Only (Weight 0.0)
# =====================================================================

def test_interleaver_evidence_provenance_only_zero_weight():
    """
    MANDATORY REQUIREMENT 2:
    EvidenceTrace.interleaver must preserve stage status, telemetry, and provenance,
    but have weight_interleaver = 0.0 and zero influence on candidate raw score / ranking.
    """
    cfg = settings.HYPOTHESIS_SCORING
    assert getattr(cfg, "weight_interleaver", 0.0) == 0.0

    # Candidate 1: Interleaver executed and SUCCESS
    c1 = create_candidate(modulation="BPSK", symbol_rate=1000.0, interleaver_config="block_4x8")
    c1.evidence.ml = _make_ml_evidence("BPSK", 0.85)
    c1.evidence.interleaver = create_interleaver_evidence(
        status=EvidenceStatus.AVAILABLE,
        score=1.0,  # Even if a score is present, weight is 0.0
        details={"rows": 4, "cols": 8, "scheme": "block"},
    )

    # Candidate 2: Same evidence, but interleaver is NOT_EVALUATED
    c2 = create_candidate(modulation="BPSK", symbol_rate=1000.0, interleaver_config="none")
    c2.evidence.ml = _make_ml_evidence("BPSK", 0.85)
    c2.evidence.interleaver = create_interleaver_evidence(
        status=EvidenceStatus.NOT_EVALUATED,
        score=None,
    )

    # Candidate 3: Same evidence, but interleaver is FAILED
    c3 = create_candidate(modulation="BPSK", symbol_rate=1000.0, interleaver_config="block_4x8")
    c3.evidence.ml = _make_ml_evidence("BPSK", 0.85)
    c3.evidence.interleaver = create_interleaver_evidence(
        status=EvidenceStatus.FAILED,
        score=0.0,
        details={"failure_reason": "Block length mismatch"},
    )

    score1 = calculate_raw_score(c1, cfg)
    score2 = calculate_raw_score(c2, cfg)
    score3 = calculate_raw_score(c3, cfg)

    # With weight_interleaver == 0.0, all three must have identical raw scores
    assert score1 == pytest.approx(score2, abs=1e-6)
    assert score2 == pytest.approx(score3, abs=1e-6)

    # Telemetry and provenance are nonetheless preserved
    assert c1.evidence.interleaver.status == EvidenceStatus.AVAILABLE
    assert c1.evidence.interleaver.details["rows"] == 4
    assert c3.evidence.interleaver.status == EvidenceStatus.FAILED
    assert "Block length mismatch" in c3.evidence.interleaver.details["failure_reason"]


# =====================================================================
# 4. Mandatory Requirement 3: Canonical Exception Handling & Isolation
# =====================================================================

def test_exception_isolation_canonical_evidence_path():
    """
    MANDATORY REQUIREMENT 3:
    If one candidate causes an unexpected decoder exception, it must:
      - Construct a well-formed DecoderResult(success=False)
      - Pass through update_hypothesis_from_decoder_result() / decoder_result_to_evidence_trace()
      - NOT manually mutate candidate evidence inside exception handlers
      - NOT crash the evaluation of other candidates
    """
    cand_good = create_candidate(modulation="BPSK", symbol_rate=1000.0)
    cand_exploding = create_candidate(modulation="QPSK", symbol_rate=2000.0)

    # Custom chain where QPSK raises an unhandled RuntimeError
    def exploding_handler(ctx: DecoderContext) -> DemodulationResult:
        if ctx.modulation == "QPSK":
            raise RuntimeError("Corrupted hardware demodulator state buffer!")
        return DemodulationResult.create_success(constellation_score=0.88)

    chain = DecoderChain(demod_stage=DemodulationAdapter(handler=exploding_handler))
    engine = HypothesisEngine(decoder_chain=chain)

    iq = _make_synthetic_iq(200)
    res = engine.execute_and_rank(iq=iq, sample_rate=10000.0, candidates=[cand_good, cand_exploding])

    # Search completed successfully with both candidates evaluated
    assert res.total_evaluated == 2
    assert len(res.ranked_hypotheses) == 2

    # The exploding candidate has canonical failed DecoderResult
    dec_exploding = res.decoder_results[cand_exploding.id]
    assert dec_exploding.success is False
    assert "Corrupted hardware demodulator state buffer!" in dec_exploding.failure_reason

    # Evidence trace was updated through the canonical path
    cand_exploding_evaluated = next(h for h in res.ranked_hypotheses if h.id == cand_exploding.id)
    assert cand_exploding_evaluated.evidence.timing.status == EvidenceStatus.FAILED
    assert cand_exploding_evaluated.evidence.constellation.status == EvidenceStatus.NOT_EVALUATED
    assert cand_exploding_evaluated.evidence.fec.status == EvidenceStatus.NOT_EVALUATED
    assert cand_exploding_evaluated.evidence.bitstream.status == EvidenceStatus.NOT_EVALUATED

    # The good candidate was evaluated without disruption
    cand_good_evaluated = next(h for h in res.ranked_hypotheses if h.id == cand_good.id)
    assert cand_good_evaluated.evidence.constellation.status == EvidenceStatus.AVAILABLE
    assert cand_good_evaluated.evidence.constellation.score == pytest.approx(0.88, abs=1e-4)


# =====================================================================
# 5. Mandatory Additional Tests: Scenario A & Scenario B
# =====================================================================

def test_scenario_a_validated_candidate_is_best():
    """
    MANDATORY SCENARIO A:
    Candidate A: BPSK, 1000 baud, conv_r1/2_k7, CRC-16 -> VALIDATED
    Candidate B: QPSK, 1000 baud -> FAILED / unsupported
    Candidate C: BPSK, 2000 baud -> FAILED / poor decode

    Verify:
      best_hypothesis == A
      validated_decode == A
    """
    cand_a = create_candidate(
        modulation="BPSK",
        symbol_rate=1000.0,
        fec_config="conv_r1/2_k7",
        sync_assumptions={"crc_scheme": "CRC-16-CCITT"},
    )
    cand_a.evidence.ml = _make_ml_evidence("BPSK", 0.85)

    cand_b = create_candidate(modulation="QPSK", symbol_rate=1000.0)
    cand_b.evidence.ml = _make_ml_evidence("QPSK", 0.10)

    cand_c = create_candidate(modulation="BPSK", symbol_rate=2000.0)
    cand_c.evidence.ml = _make_ml_evidence("BPSK", 0.05)

    # Mock chain handler:
    # A validates successfully; B is unsupported; C fails validation
    class ScenarioAChain(DecoderChain):
        def decode(self, request: DecoderRequest) -> DecoderResult:
            cid = request.hypothesis.id
            if request.hypothesis.modulation == "BPSK" and request.hypothesis.symbolRate == 1000.0:
                return DecoderResult(
                    success=True,
                    hypothesis_id=cid,
                    synchronization=SynchronizationResult.create_success(timing_score=0.95),
                    demodulation=DemodulationResult.create_success(constellation_score=0.90),
                    interleaver=InterleaverResult.not_evaluated(),
                    fec=FECResult.create_success(fec_score=1.0, error_rate=0.0),
                    validation=ValidationResult.create_success(validation_score=1.0, is_valid=True),
                )
            elif request.hypothesis.modulation == "QPSK":
                return DecoderResult(
                    success=False,
                    hypothesis_id=cid,
                    failure_reason="Unsupported modulation",
                    synchronization=SynchronizationResult.create_success(timing_score=0.50),
                    demodulation=DemodulationResult.not_supported(failure_reason="Unsupported"),
                    interleaver=InterleaverResult.not_evaluated(),
                    fec=FECResult.not_evaluated(),
                    validation=ValidationResult.not_evaluated(),
                )
            else:
                return DecoderResult(
                    success=False,
                    hypothesis_id=cid,
                    failure_reason="Validation CRC error",
                    synchronization=SynchronizationResult.create_success(timing_score=0.30),
                    demodulation=DemodulationResult.create_success(constellation_score=0.30),
                    interleaver=InterleaverResult.not_evaluated(),
                    fec=FECResult.failed(failure_reason="Viterbi failure"),
                    validation=ValidationResult.failed(failure_reason="CRC mismatch"),
                )

    engine = HypothesisEngine(decoder_chain=ScenarioAChain())
    iq = _make_synthetic_iq(200)
    res = engine.execute_and_rank(
        iq=iq,
        sample_rate=10000.0,
        candidates=[cand_a, cand_b, cand_c],
    )

    assert res.total_evaluated == 3
    assert res.successful_decodes_count == 1
    assert res.failed_decodes_count == 2

    # Verify best_hypothesis == A and validated_decode == A
    assert res.best_hypothesis is not None
    assert res.best_hypothesis.id == cand_a.id

    assert res.validated_decode is not None
    assert res.validated_decode.id == cand_a.id

    assert res.best_hypothesis.id == res.validated_decode.id


def test_scenario_b_best_hypothesis_is_not_validated():
    """
    MANDATORY SCENARIO B & REQUIREMENT 1:
    Candidate A: highest confidence -> incomplete/failed decode
    Candidate B: lower confidence -> complete validated decode
    Candidate C: failed/unsupported

    Verify:
      best_hypothesis == A
      validated_decode == B
      best_hypothesis != validated_decode
    """
    # Candidate A: Dominant ML score (0.95), but CRC decode FAILED
    cand_a = create_candidate(modulation="BPSK", symbol_rate=1000.0)
    cand_a.evidence.ml = _make_ml_evidence("BPSK", 0.98)

    # Candidate B: Lower ML score (0.20), but decode SUCCESS (CRC valid)
    cand_b = create_candidate(modulation="QPSK", symbol_rate=1000.0, fec_config="conv_r1/2_k7")
    cand_b.evidence.ml = _make_ml_evidence("QPSK", 0.20)

    # Candidate C: Very low ML score (0.05), unsupported/failed
    cand_c = create_candidate(modulation="FSK", symbol_rate=500.0)
    cand_c.evidence.ml = _make_ml_evidence("FSK", 0.05)

    class ScenarioBChain(DecoderChain):
        def decode(self, request: DecoderRequest) -> DecoderResult:
            cid = request.hypothesis.id
            if request.hypothesis.modulation == "BPSK":
                # High physical quality (sync=0.95, demod=0.95), but unconfigured CRC or CRC mismatch
                return DecoderResult(
                    success=False,
                    hypothesis_id=cid,
                    failure_reason="CRC checksum failed",
                    synchronization=SynchronizationResult.create_success(timing_score=0.95),
                    demodulation=DemodulationResult.create_success(constellation_score=0.95),
                    interleaver=InterleaverResult.not_evaluated(),
                    fec=FECResult.not_evaluated(),
                    validation=ValidationResult.not_supported(failure_reason="No CRC specified"),
                )
            elif request.hypothesis.modulation == "QPSK":
                # Validated decode!
                return DecoderResult(
                    success=True,
                    hypothesis_id=cid,
                    synchronization=SynchronizationResult.create_success(timing_score=0.50),
                    demodulation=DemodulationResult.create_success(constellation_score=0.50),
                    interleaver=InterleaverResult.not_evaluated(),
                    fec=FECResult.create_success(fec_score=0.80),
                    validation=ValidationResult.create_success(validation_score=1.0, is_valid=True),
                )
            else:
                return DecoderResult(
                    success=False,
                    hypothesis_id=cid,
                    failure_reason="Unsupported FSK",
                    synchronization=SynchronizationResult.failed(failure_reason="Sync failed"),
                    demodulation=DemodulationResult.not_supported(),
                    interleaver=InterleaverResult.not_evaluated(),
                    fec=FECResult.not_evaluated(),
                    validation=ValidationResult.not_evaluated(),
                )

    engine = HypothesisEngine(decoder_chain=ScenarioBChain())
    iq = _make_synthetic_iq(200)
    res = engine.execute_and_rank(
        iq=iq,
        sample_rate=10000.0,
        candidates=[cand_a, cand_b, cand_c],
    )

    # Candidate A has much higher raw score and confidence than Candidate B:
    # A has ml=0.98, timing=0.95, constellation=0.95 -> raw score ~ 0.96
    # B has ml=0.20, timing=0.50, constellation=0.50, fec=0.80, val=1.0 -> raw score ~ 0.50
    # Candidate A ranks #1, Candidate B ranks #2
    assert res.ranked_hypotheses[0].id == cand_a.id
    assert res.ranked_hypotheses[1].id == cand_b.id

    # VERIFY MANDATORY DISTINCTION:
    # best_hypothesis must be A (highest confidence)
    assert res.best_hypothesis.id == cand_a.id

    # validated_decode must be B (highest-ranked candidate where DecoderResult.success == True)
    assert res.validated_decode.id == cand_b.id

    # They MUST NOT be identical in this scenario
    assert res.best_hypothesis.id != res.validated_decode.id


def test_no_validated_decode_when_all_fail():
    """Verifies that validated_decode is None when no candidate achieves success=True."""
    cand1 = create_candidate(modulation="BPSK", symbol_rate=1000.0)
    cand2 = create_candidate(modulation="QPSK", symbol_rate=2000.0)

    class AllFailChain(DecoderChain):
        def decode(self, request: DecoderRequest) -> DecoderResult:
            return DecoderResult(
                success=False,
                hypothesis_id=request.hypothesis.id,
                failure_reason="Forced failure",
                synchronization=SynchronizationResult.create_success(timing_score=0.8),
                demodulation=DemodulationResult.create_success(constellation_score=0.8),
                interleaver=InterleaverResult.not_evaluated(),
                fec=FECResult.not_evaluated(),
                validation=ValidationResult.failed(failure_reason="Failed"),
            )

    engine = HypothesisEngine(decoder_chain=AllFailChain())
    iq = _make_synthetic_iq(100)
    res = engine.execute_and_rank(iq=iq, sample_rate=10000.0, candidates=[cand1, cand2])

    assert res.best_hypothesis is not None
    assert res.validated_decode is None
    assert res.successful_decodes_count == 0
    assert res.failed_decodes_count == 2


# =====================================================================
# 6. Confidence Normalization, Deduplication & Determinism
# =====================================================================

def test_confidence_normalization_and_sum_to_one():
    """Verifies that all returned candidate confidences are in [0, 1] and sum to 1.0."""
    cands = [
        create_candidate(modulation="BPSK", symbol_rate=1000.0),
        create_candidate(modulation="QPSK", symbol_rate=2000.0),
        create_candidate(modulation="8PSK", symbol_rate=3000.0),
    ]
    for i, c in enumerate(cands):
        c.evidence.ml = _make_ml_evidence(c.modulation, 0.3 + 0.2 * i)

    engine = HypothesisEngine()
    iq = _make_synthetic_iq(200)
    res = engine.execute_and_rank(iq=iq, sample_rate=10000.0, candidates=cands)

    confs = [c.confidenceScore for c in res.ranked_hypotheses]
    for conf in confs:
        assert 0.0 <= conf <= 1.0
    assert sum(confs) == pytest.approx(1.0, abs=1e-5)


def test_deduplication_preserves_distinct_collapses_identical():
    """Verifies that identical configurations are deduplicated, but distinct ones preserved."""
    c1 = create_candidate(modulation="BPSK", symbol_rate=1000.0)
    c2 = create_candidate(modulation="BPSK", symbol_rate=1000.0)  # duplicate
    c3 = create_candidate(modulation="BPSK", symbol_rate=2000.0)  # distinct rate

    engine = HypothesisEngine()
    iq = _make_synthetic_iq(100)
    res = engine.execute_and_rank(
        iq=iq,
        sample_rate=10000.0,
        candidates=[c1, c2, c3],
        deduplicate=True,
    )

    # 3 candidates evaluated, but deduplication reduces to 2 distinct configurations
    assert res.total_evaluated == 3
    assert len(res.ranked_hypotheses) == 2


def test_deterministic_execution():
    """Verifies that running execute_and_rank multiple times with identical inputs produces identical results."""
    c1 = create_candidate(modulation="BPSK", symbol_rate=1000.0)
    c2 = create_candidate(modulation="QPSK", symbol_rate=2000.0)
    iq = _make_synthetic_iq(200)

    engine = HypothesisEngine()
    res1 = engine.execute_and_rank(iq=iq, sample_rate=10000.0, candidates=[c1, c2])
    res2 = engine.execute_and_rank(iq=iq, sample_rate=10000.0, candidates=[c1, c2])

    assert res1.best_hypothesis.id == res2.best_hypothesis.id
    assert [h.confidenceScore for h in res1.ranked_hypotheses] == [h.confidenceScore for h in res2.ranked_hypotheses]


def test_result_serialization_to_dict():
    """Verifies that HypothesisEngineResult serializes to a valid dictionary."""
    c1 = create_candidate(modulation="BPSK", symbol_rate=1000.0)
    engine = HypothesisEngine()
    iq = _make_synthetic_iq(100)
    res = engine.execute_and_rank(iq=iq, sample_rate=10000.0, candidates=[c1])

    d = res.to_dict()
    assert isinstance(d, dict)
    assert d["total_evaluated"] == 1
    assert "best_hypothesis_id" in d
    assert "best_hypothesis_confidence" in d
    assert "has_validated_decode" in d
    assert "ranked_hypotheses" in d
    assert "decoder_results" in d


# =====================================================================
# 7. Synthetic End-to-End Signal Integration
# =====================================================================

def test_synthetic_end_to_end_pipeline():
    """
    FULL PIPELINE TEST:
    P5.2 candidate creation -> P5.3 DecoderChain -> P5.4 genuine Viterbi & CRC validation
    -> evidence aggregation -> ranking -> best_hypothesis + validated_decode.
    """
    # 1. Synthesize a clean BPSK frame with genuine CRC-16-CCITT and K=7 Rate 1/2 Viterbi encoding
    payload = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
    framed_bits = append_crc_to_bits(payload, crc_name="CRC-16-CCITT")  # 16 + 16 = 32 bits

    encoded_bits = encode_convolutional_r12_k7(framed_bits)  # 64 encoded bits

    # Modulate to BPSK symbols: bit 1 -> +1.0, bit 0 -> -1.0 (matching demodulate_psk decision boundary real > 0)
    bpsk_symbols = np.where(encoded_bits == 1, 1.0, -1.0).astype(np.float32)

    # 1 sample per symbol
    i_signal = bpsk_symbols
    q_signal = np.zeros_like(i_signal)
    iq = np.vstack([i_signal, q_signal]).astype(np.float32)

    sample_rate = 1000.0
    baud = 1000.0

    # Candidate 1: Matching candidate (BPSK, 1000 baud, conv_r1/2_k7, CRC-16-CCITT)
    cand_match = create_candidate(
        modulation="BPSK",
        symbol_rate=baud,
        fec_config="conv_r1/2_k7",
        sync_assumptions={"crc_scheme": "CRC-16-CCITT", "matched_filter": "none"},
    )
    cand_match.evidence.ml = _make_ml_evidence("BPSK", 0.90)

    # Candidate 2: Wrong modulation candidate (QPSK)
    cand_wrong_mod = create_candidate(
        modulation="QPSK",
        symbol_rate=baud,
        fec_config="conv_r1/2_k7",
        sync_assumptions={"crc_scheme": "CRC-16-CCITT"},
    )
    cand_wrong_mod.evidence.ml = _make_ml_evidence("QPSK", 0.10)

    # Execute engine with genuine DecoderChain (no mocks!)
    engine = HypothesisEngine()
    result = engine.execute_and_rank(
        iq=iq,
        sample_rate=sample_rate,
        candidates=[cand_match, cand_wrong_mod],
    )

    # Candidate 1 must succeed end-to-end through Viterbi and CRC!
    assert result.successful_decodes_count == 1
    assert result.failed_decodes_count == 1

    # Candidate 1 must be both best_hypothesis and validated_decode
    assert result.best_hypothesis is not None
    assert result.best_hypothesis.id == cand_match.id

    assert result.validated_decode is not None
    assert result.validated_decode.id == cand_match.id

    # The payload reconstructed by candidate 1 should match original payload
    match_dec_res = result.decoder_results[cand_match.id]
    assert match_dec_res.success is True
    assert match_dec_res.validation.is_valid is True
    assert match_dec_res.validation.status == DecoderStageStatus.SUCCESS
    assert match_dec_res.bits is not None
    assert np.array_equal(match_dec_res.bits[:len(payload)], payload)
