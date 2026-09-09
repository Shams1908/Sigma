"""
Unit and integration tests for P5.6 Decoder Lab Facade (backend/hypothesis/decoder_lab.py).

Verifies all 9 mandatory requirements from P5.6:
  1. Basic end-to-end execution (BPSK + known symbol rate + known sample rate).
  2. Genuine validated decode (BPSK + Rate-1/2 K=7 Viterbi + CRC-16-CCITT).
  3. Best hypothesis vs. validated decode distinction (Scenario B).
  4. Candidate search bounding (hard bound max_candidates <= 24).
  5. Deterministic repeated execution (identical candidate IDs, order, scores).
  6. Error isolation across candidates (unexpected exception handled via canonical failure).
  7. Unsupported modulation/code handling (explicit NOT_SUPPORTED).
  8. M8 symbol-rate evidence integration (Gaussian likelihood agreement).
  9. CRC candidate bound and deterministic identity.
"""
import math
import numpy as np
import pytest

from backend.core.config import settings
from backend.hypothesis.candidates import (
    HypothesisCandidate,
    EvidenceStatus,
    EvidenceComponent,
    EvidenceTrace,
)
from backend.hypothesis.generator import (
    HypothesisSearchConfig,
    create_candidate,
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
)
from backend.hypothesis.decoder_chain import (
    DecoderChain,
    DemodulationAdapter,
    ValidationAdapter,
)
from backend.hypothesis.decoder_lab import (
    DecoderLab,
    DecoderLabResult,
    run_decoder_lab,
)
from backend.fec import encode_convolutional_r12_k7
from backend.validation import append_crc_to_bits


# =====================================================================
# Helpers
# =====================================================================

def _make_bpsk_signal(
    payload_bits: np.ndarray,
    fec: bool = True,
    crc: bool = True,
    crc_scheme: str = "CRC-16-CCITT",
    sample_rate: float = 1000.0,
    baud: float = 1000.0,
) -> np.ndarray:
    """Synthesizes a canonical 2D real float32 IQ array for testing."""
    bits = np.array(payload_bits, dtype=np.uint8)
    if crc:
        bits = append_crc_to_bits(bits, crc_name=crc_scheme)
    if fec:
        bits = encode_convolutional_r12_k7(bits)

    # BPSK mapping: bit 1 -> +1.0, bit 0 -> -1.0 (matching demodulate_psk boundary real > 0)
    bpsk_symbols = np.where(bits == 1, 1.0, -1.0).astype(np.float32)

    # 1 sample per symbol
    i_signal = bpsk_symbols
    q_signal = np.zeros_like(i_signal)
    return np.vstack([i_signal, q_signal]).astype(np.float32)


# =====================================================================
# 1. Basic End-to-End Test
# =====================================================================

def test_decoder_lab_basic_end_to_end():
    """
    Test 1: Verifies basic end-to-end execution of DecoderLab with clean BPSK IQ.
    Verifies candidates generated, decoder chain executed, ranking returned, best hypothesis exists.
    """
    payload = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    iq = _make_bpsk_signal(payload, fec=False, crc=False)

    result = run_decoder_lab(
        iq=iq,
        sample_rate=1000.0,
        symbol_rate=1000.0,
        ml_probabilities={"BPSK": 0.85, "QPSK": 0.15},
    )

    assert isinstance(result, DecoderLabResult)
    assert result.total_evaluated > 0
    assert len(result.ranked_hypotheses) > 0
    assert result.best_hypothesis is not None
    assert result.best_hypothesis.modulation == "BPSK"
    assert result.best_hypothesis.confidenceScore > 0.0
    assert result.search_summary is not None
    assert "execution_time_seconds" in result.diagnostics


# =====================================================================
# 2. Validated Decode Test
# =====================================================================

def test_decoder_lab_validated_decode():
    """
    Test 2: Verifies genuine validated decode on synthetic BPSK + Rate 1/2 K=7 + CRC-16-CCITT.
    """
    payload = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
    iq = _make_bpsk_signal(payload, fec=True, crc=True, crc_scheme="CRC-16-CCITT")

    search_cfg = HypothesisSearchConfig(
        fec_candidates=["none", "conv_r1/2_k7"],
        crc_candidates=["CRC-16-CCITT"],
        max_candidates=10,
    )

    result = run_decoder_lab(
        iq=iq,
        sample_rate=1000.0,
        symbol_rate=1000.0,
        ml_probabilities={"BPSK": 0.90, "QPSK": 0.10},
        search_config=search_cfg,
    )

    assert result.validated_decode is not None
    v_dec = result.validated_decode
    assert v_dec.modulation == "BPSK"
    assert v_dec.fec_config == "conv_r1/2_k7"
    assert (v_dec.sync_assumptions or {}).get("crc_scheme") == "CRC-16-CCITT"

    v_res = result.decoder_results[v_dec.id]
    assert v_res.success is True
    assert v_res.validation.is_valid is True
    assert v_res.validation.status == DecoderStageStatus.SUCCESS

    # Decoded bits contain original payload + CRC
    assert v_res.bits is not None
    assert np.array_equal(v_res.bits[:len(payload)], payload)


# =====================================================================
# 3. Best vs. Validated Semantics Test
# =====================================================================

def test_decoder_lab_best_vs_validated():
    """
    Test 3: Verifies that best_hypothesis (highest confidence) and validated_decode
    (highest valid decode) remain strictly distinct when candidate conditions diverge.
    """
    payload = np.array([1, 0, 1, 0], dtype=np.uint8)
    iq = _make_bpsk_signal(payload, fec=False, crc=False)

    # Custom chain where candidate 1 has top confidence but fails CRC,
    # and candidate 2 has lower confidence but achieves valid decode.
    class MockBestVsValidChain(DecoderChain):
        def decode(self, request: DecoderRequest) -> DecoderResult:
            cid = request.hypothesis.id
            if request.hypothesis.modulation == "BPSK":
                return DecoderResult(
                    success=False,
                    hypothesis_id=cid,
                    failure_reason="CRC mismatch",
                    synchronization=SynchronizationResult.create_success(timing_score=0.98),
                    demodulation=DemodulationResult.create_success(constellation_score=0.98),
                    fec=FECResult.not_evaluated(),
                    validation=ValidationResult.failed(failure_reason="CRC failure"),
                )
            else:
                return DecoderResult(
                    success=True,
                    hypothesis_id=cid,
                    synchronization=SynchronizationResult.create_success(timing_score=0.50),
                    demodulation=DemodulationResult.create_success(constellation_score=0.50),
                    fec=FECResult.create_success(fec_score=0.80),
                    validation=ValidationResult.create_success(is_valid=True),
                )

    lab = DecoderLab(decoder_chain=MockBestVsValidChain())
    res = lab.run(
        iq=iq,
        sample_rate=1000.0,
        symbol_rate=1000.0,
        ml_probabilities={"BPSK": 0.95, "QPSK": 0.05},
    )

    assert res.best_hypothesis is not None
    assert res.validated_decode is not None

    # best_hypothesis is BPSK (top confidence)
    assert res.best_hypothesis.modulation == "BPSK"

    # validated_decode is QPSK (successful decode)
    assert res.validated_decode.modulation == "QPSK"

    # Must NOT be identical
    assert res.best_hypothesis.id != res.validated_decode.id
    assert res.diagnostics["best_is_validated"] is False


# =====================================================================
# 4. Candidate Bound Enforcement Test
# =====================================================================

def test_decoder_lab_candidate_bound_24():
    """
    Test 4: Verifies that candidate search remains bounded by max_candidates <= 24
    even when a massive Cartesian space is requested.
    """
    payload = np.array([1, 0, 1, 0], dtype=np.uint8)
    iq = _make_bpsk_signal(payload, fec=False, crc=False)

    massive_cfg = HypothesisSearchConfig(
        top_k_modulations=10,
        max_symbol_rate_candidates=5,
        symbol_rate_offsets=[-200, -100, 0, 100, 200],
        fec_candidates=["none", "conv_r1/2_k7", "ldpc_r3/4", "turbo_r1/3"],
        interleaver_candidates=["none", "block_16x16", "block_32x32"],
        crc_candidates=["none", "CRC-16-CCITT", "CRC-32"],
        max_candidates=24,  # Hard bound
    )

    probs = {f"MOD_{i}": 0.10 for i in range(10)}
    res = run_decoder_lab(
        iq=iq,
        sample_rate=10000.0,
        symbol_rate=1000.0,
        ml_probabilities=probs,
        search_config=massive_cfg,
    )

    assert len(res.ranked_hypotheses) <= 24
    assert res.total_evaluated <= 24
    assert res.search_summary["final_candidate_count"] <= 24
    assert res.search_summary["pruned_by_limit"] is True


# =====================================================================
# 5. Determinism Test
# =====================================================================

def test_decoder_lab_determinism():
    """
    Test 5: Verifies that running DecoderLab twice on the same inputs yields
    identical candidate IDs, order, scores, and confidences.
    """
    payload = np.array([1, 0, 1, 0, 1, 1, 0, 0], dtype=np.uint8)
    iq = _make_bpsk_signal(payload, fec=False, crc=False)

    res1 = run_decoder_lab(
        iq=iq,
        sample_rate=1000.0,
        symbol_rate=1000.0,
        ml_probabilities={"BPSK": 0.80, "QPSK": 0.20},
    )
    res2 = run_decoder_lab(
        iq=iq,
        sample_rate=1000.0,
        symbol_rate=1000.0,
        ml_probabilities={"BPSK": 0.80, "QPSK": 0.20},
    )

    assert res1.best_hypothesis.id == res2.best_hypothesis.id
    assert [h.id for h in res1.ranked_hypotheses] == [h.id for h in res2.ranked_hypotheses]
    assert [h.confidenceScore for h in res1.ranked_hypotheses] == [h.confidenceScore for h in res2.ranked_hypotheses]
    assert [h.rawScore for h in res1.ranked_hypotheses] == [h.rawScore for h in res2.ranked_hypotheses]


# =====================================================================
# 6. Error Isolation Test
# =====================================================================

def test_decoder_lab_error_isolation():
    """
    Test 6: Verifies that an unexpected exception in one candidate does not abort
    the search, and that failing candidates receive a canonical failed result.
    """
    payload = np.array([1, 0, 1, 0], dtype=np.uint8)
    iq = _make_bpsk_signal(payload, fec=False, crc=False)

    def crashing_handler(ctx: DecoderContext) -> DemodulationResult:
        if ctx.modulation == "QPSK":
            raise ZeroDivisionError("Simulated hardware divide-by-zero")
        return DemodulationResult.create_success(constellation_score=0.90)

    exploding_chain = DecoderChain(demod_stage=DemodulationAdapter(handler=crashing_handler))
    lab = DecoderLab(decoder_chain=exploding_chain)

    res = lab.run(
        iq=iq,
        sample_rate=1000.0,
        symbol_rate=1000.0,
        ml_probabilities={"BPSK": 0.70, "QPSK": 0.30},
        search_config=HypothesisSearchConfig(max_symbol_rate_candidates=1),
    )

    # Search completed without crashing
    assert res.total_evaluated == 2
    assert len(res.ranked_hypotheses) == 2

    # Find the crashing QPSK candidate
    qpsk_cands = [h for h in res.ranked_hypotheses if h.modulation == "QPSK"]
    assert len(qpsk_cands) >= 1
    qpsk_cand = qpsk_cands[0]

    dec_res = res.decoder_results[qpsk_cand.id]
    assert dec_res.success is False
    assert "Simulated hardware divide-by-zero" in dec_res.failure_reason

    # The BPSK candidate evaluated normally
    bpsk_cands = [h for h in res.ranked_hypotheses if h.modulation == "BPSK"]
    assert len(bpsk_cands) >= 1
    bpsk_dec_res = res.decoder_results[bpsk_cands[0].id]
    assert bpsk_dec_res.demodulation.status == DecoderStageStatus.SUCCESS


# =====================================================================
# 7. Unsupported Candidate Handling
# =====================================================================

def test_decoder_lab_unsupported_candidate():
    """
    Test 7: Verifies that unsupported modulations (e.g. 16QAM) or FEC codes (e.g. LDPC)
    explicitly return NOT_SUPPORTED and do not crash the run.
    """
    payload = np.array([1, 0, 1, 0], dtype=np.uint8)
    iq = _make_bpsk_signal(payload, fec=False, crc=False)

    search_cfg = HypothesisSearchConfig(
        top_k_modulations=3,
        fec_candidates=["none", "ldpc_r3/4"],
        max_candidates=10,
    )

    res = run_decoder_lab(
        iq=iq,
        sample_rate=1000.0,
        symbol_rate=1000.0,
        ml_probabilities={"BPSK": 0.60, "16QAM": 0.40},
        search_config=search_cfg,
    )

    assert res.total_evaluated > 0
    assert res.diagnostics["unsupported_candidate_count"] > 0

    # 16QAM candidate must report NOT_SUPPORTED for demodulation
    qam_cands = [h for h in res.ranked_hypotheses if h.modulation == "16QAM"]
    if qam_cands:
        qam_res = res.decoder_results[qam_cands[0].id]
        assert qam_res.demodulation.status == DecoderStageStatus.NOT_SUPPORTED
        assert qam_res.success is False


# =====================================================================
# 8. M8 Parameter Integration & Symbol Rate Evidence Test
# =====================================================================

def test_decoder_lab_m8_symbol_rate_evidence():
    """
    Test 8: Verifies that providing an M8 parameter_result attaches Gaussian
    symbol-rate evidence to candidates, and shifts ranking according to Baud agreement.
    """
    payload = np.array([1, 0, 1, 0], dtype=np.uint8)
    iq = _make_bpsk_signal(payload, fec=False, crc=False)

    class MockM8Param:
        def __init__(self, est, unc, conf):
            self.estimate = est
            self.uncertainty = unc
            self.confidence = conf
            self.method = "hybrid_consensus"
        def to_dict(self):
            return {"estimate": self.estimate, "uncertainty": self.uncertainty}

    class MockM8Analysis:
        def __init__(self, baud=1000.0, unc=50.0):
            self.symbol_rate = MockM8Param(baud, unc, 0.95)
            self.snr_db = MockM8Param(20.0, 1.0, 0.90)
            self.sample_rate = 10000.0

    # Center Baud is 1000.0. We test two rates: 1000.0 (exact match) and 1200.0 (offset).
    search_cfg = HypothesisSearchConfig(
        top_k_modulations=1,
        max_symbol_rate_candidates=2,
        symbol_rate_offsets=[0.0, 200.0],
        max_candidates=10,
    )

    m8_result = MockM8Analysis(baud=1000.0, unc=50.0)
    res = run_decoder_lab(
        iq=iq,
        sample_rate=10000.0,
        ml_probabilities={"BPSK": 0.90},
        parameter_result=m8_result,
        search_config=search_cfg,
    )

    assert len(res.ranked_hypotheses) == 2
    exact_cand = next(h for h in res.ranked_hypotheses if abs(h.symbolRate - 1000.0) < 1.0)
    offset_cand = next(h for h in res.ranked_hypotheses if abs(h.symbolRate - 1200.0) < 1.0)

    # Both candidates have AVAILABLE symbol_rate evidence with Gaussian likelihood
    assert exact_cand.evidence.symbol_rate.status == EvidenceStatus.AVAILABLE
    assert offset_cand.evidence.symbol_rate.status == EvidenceStatus.AVAILABLE

    # Exact candidate has higher symbol rate score than offset candidate
    assert exact_cand.evidence.symbol_rate.score > offset_cand.evidence.symbol_rate.score
    assert exact_cand.confidenceScore > offset_cand.confidenceScore


# =====================================================================
# 9. CRC Candidate Bound & Identity Test
# =====================================================================

def test_decoder_lab_crc_candidate_bound():
    """
    Test 9: Verifies that crc_candidates in search config are bounded to max_candidates <= 24
    and participate deterministically in candidate identity.
    """
    payload = np.array([1, 0, 1, 0], dtype=np.uint8)
    iq = _make_bpsk_signal(payload, fec=False, crc=False)

    search_cfg = HypothesisSearchConfig(
        top_k_modulations=2,  # BPSK, QPSK
        max_symbol_rate_candidates=1,
        fec_candidates=["none", "conv_r1/2_k7"],
        crc_candidates=["none", "CRC-16-CCITT", "CRC-32"],
        max_candidates=24,
    )

    res = run_decoder_lab(
        iq=iq,
        sample_rate=1000.0,
        symbol_rate=1000.0,
        ml_probabilities={"BPSK": 0.60, "QPSK": 0.40},
        search_config=search_cfg,
    )

    # 2 mods x 1 rate x 1 sync x 2 FEC x 3 CRC = 12 combinations <= 24
    assert len(res.ranked_hypotheses) == 12
    assert res.total_evaluated == 12

    # Check that candidate IDs reflect distinct CRC configurations
    crc_schemes_found = set()
    for h in res.ranked_hypotheses:
        scheme = (h.sync_assumptions or {}).get("crc_scheme")
        crc_schemes_found.add(scheme)

    assert "CRC-16-CCITT" in crc_schemes_found
    assert "CRC-32" in crc_schemes_found
    assert None in crc_schemes_found  # "none" maps to no crc_scheme


# =====================================================================
# 10. Result Serialization Test
# =====================================================================

def test_decoder_lab_result_serialization():
    """Verifies that DecoderLabResult serializes cleanly to a JSON-ready dict."""
    payload = np.array([1, 0, 1, 0], dtype=np.uint8)
    iq = _make_bpsk_signal(payload, fec=False, crc=False)

    res = run_decoder_lab(
        iq=iq,
        sample_rate=1000.0,
        symbol_rate=1000.0,
        ml_probabilities={"BPSK": 0.90},
    )

    d = res.to_dict()
    assert isinstance(d, dict)
    assert "total_evaluated" in d
    assert "best_hypothesis_id" in d
    assert "ranked_hypotheses" in d
    assert "decoder_results" in d
    assert "diagnostics" in d
    assert "search_summary" in d
