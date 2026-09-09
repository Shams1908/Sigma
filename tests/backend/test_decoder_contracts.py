"""
Unit tests for P5.1 Decoder Contracts & Architecture.
Tests:
  - DecoderRequest canonical IQ validation and format conversions.
  - Stage-level result contracts (Synchronization, Demodulation, Interleaver, FEC, Validation).
  - DecoderResult structure, serialization, and explicit failure semantics.
  - DecoderContext lightweight state tracking.
  - DecoderEngine interface and ScaffoldDecoderEngine non-fabrication.
  - Integration adapter with HypothesisCandidate, EvidenceTrace, calculate_raw_score, and rank_hypotheses.
"""
import json
import math
import numpy as np
import pytest

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
)
from backend.hypothesis.ranking import rank_hypotheses
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


def _make_candidate(
    modulation: str = "QPSK",
    symbol_rate: float = 10000.0,
    cand_id: str = "hyp_test_01",
) -> HypothesisCandidate:
    """Helper to create a standard candidate hypothesis."""
    return HypothesisCandidate(
        id=cand_id,
        modulation=modulation,
        symbolRate=symbol_rate,
        fec_config="conv_r1/2_k7",
        evidence=EvidenceTrace(
            ml=create_ml_evidence(modulation, {modulation: 0.90}),
            symbol_rate=create_symbol_rate_evidence(symbol_rate, symbol_rate),
        ),
    )


# =====================================================================
# 1. DecoderRequest Tests
# =====================================================================

def test_decoder_request_valid_2d_real_iq():
    """Verifies valid construction with canonical 2D real float32 array [2, N]."""
    iq = np.ones((2, 1000), dtype=np.float32) * 0.5
    cand = _make_candidate()
    req = DecoderRequest(iq=iq, hypothesis=cand, sample_rate=100000.0)

    assert req.num_samples == 1000
    assert req.sample_rate == 100000.0
    assert req.hypothesis.modulation == "QPSK"

    # Canonical conversions
    iq_2d = req.canonical_iq_2d()
    assert iq_2d.shape == (2, 1000)
    assert iq_2d.dtype == np.float32

    iq_c = req.canonical_iq_complex()
    assert iq_c.shape == (1000,)
    assert np.iscomplexobj(iq_c)
    assert np.allclose(iq_c.real, 0.5)
    assert np.allclose(iq_c.imag, 0.5)


def test_decoder_request_valid_1d_complex_iq():
    """Verifies valid construction with canonical 1D complex array [N]."""
    iq = np.full(500, 0.707 + 0.707j, dtype=np.complex64)
    cand = _make_candidate()
    req = DecoderRequest(iq=iq, hypothesis=cand, sample_rate=50000.0)

    assert req.num_samples == 500
    iq_2d = req.canonical_iq_2d()
    assert iq_2d.shape == (2, 500)
    assert np.allclose(iq_2d[0], 0.707, atol=1e-3)
    assert np.allclose(iq_2d[1], 0.707, atol=1e-3)

    iq_c = req.canonical_iq_complex()
    assert iq_c.shape == (500,)
    assert np.iscomplexobj(iq_c)


def test_decoder_request_rejects_none_iq():
    """Verifies that None IQ is rejected."""
    cand = _make_candidate()
    with pytest.raises(ValueError, match="IQ data must not be None"):
        DecoderRequest(iq=None, hypothesis=cand, sample_rate=100000.0)


def test_decoder_request_rejects_non_numpy_iq():
    """Verifies that non-numpy IQ is rejected."""
    cand = _make_candidate()
    with pytest.raises(TypeError, match="IQ data must be a numpy.ndarray"):
        DecoderRequest(iq=[1.0, 2.0], hypothesis=cand, sample_rate=100000.0)


def test_decoder_request_rejects_empty_iq():
    """Verifies that empty IQ array is rejected."""
    cand = _make_candidate()
    with pytest.raises(ValueError, match="0 samples"):
        DecoderRequest(iq=np.array([], dtype=np.float32), hypothesis=cand, sample_rate=100000.0)


def test_decoder_request_rejects_ambiguous_1d_real_iq():
    """Verifies that 1D real IQ is rejected as ambiguous."""
    cand = _make_candidate()
    iq_1d_real = np.ones(100, dtype=np.float32)
    with pytest.raises(ValueError, match="1D real IQ array layout is ambiguous"):
        DecoderRequest(iq=iq_1d_real, hypothesis=cand, sample_rate=100000.0)


def test_decoder_request_rejects_invalid_2d_real_shape():
    """Verifies that 2D real array with shape != (2, N) is rejected."""
    cand = _make_candidate()
    iq_3x100 = np.ones((3, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="2D real IQ array must have shape \\(2, N\\)"):
        DecoderRequest(iq=iq_3x100, hypothesis=cand, sample_rate=100000.0)


def test_decoder_request_rejects_multidimensional_iq():
    """Verifies that 3D array is rejected."""
    cand = _make_candidate()
    iq_3d = np.ones((2, 2, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="Unsupported multidimensional IQ array shape"):
        DecoderRequest(iq=iq_3d, hypothesis=cand, sample_rate=100000.0)


def test_decoder_request_rejects_2d_complex_iq():
    """Verifies that 2D complex array is rejected."""
    cand = _make_candidate()
    iq_2d_c = np.ones((2, 100), dtype=np.complex64)
    with pytest.raises(ValueError, match="Complex IQ array must be 1D with shape \\(N,\\)"):
        DecoderRequest(iq=iq_2d_c, hypothesis=cand, sample_rate=100000.0)


def test_decoder_request_rejects_nan_iq():
    """Verifies that IQ containing NaNs is rejected."""
    cand = _make_candidate()
    iq = np.ones((2, 100), dtype=np.float32)
    iq[0, 50] = np.nan
    with pytest.raises(ValueError, match="contains non-finite values"):
        DecoderRequest(iq=iq, hypothesis=cand, sample_rate=100000.0)


def test_decoder_request_rejects_inf_iq():
    """Verifies that IQ containing infinities is rejected."""
    cand = _make_candidate()
    iq = np.ones((2, 100), dtype=np.float32)
    iq[1, 10] = np.inf
    with pytest.raises(ValueError, match="contains non-finite values"):
        DecoderRequest(iq=iq, hypothesis=cand, sample_rate=100000.0)


def test_decoder_request_rejects_non_positive_sample_rate():
    """Verifies that sample_rate <= 0 is rejected."""
    cand = _make_candidate()
    iq = np.ones((2, 100), dtype=np.float32)

    with pytest.raises(ValueError, match="Sample rate must be a finite positive number"):
        DecoderRequest(iq=iq, hypothesis=cand, sample_rate=0.0)

    with pytest.raises(ValueError, match="Sample rate must be a finite positive number"):
        DecoderRequest(iq=iq, hypothesis=cand, sample_rate=-48000.0)


def test_decoder_request_rejects_invalid_hypothesis():
    """Verifies that invalid hypothesis objects or missing fields are rejected."""
    iq = np.ones((2, 100), dtype=np.float32)

    # Invalid type
    with pytest.raises(TypeError, match="hypothesis must be a HypothesisCandidate instance"):
        DecoderRequest(iq=iq, hypothesis="not_a_hypothesis", sample_rate=100000.0)

    # Empty modulation
    bad_cand_mod = HypothesisCandidate(id="test", modulation="", symbolRate=1000.0)
    with pytest.raises(ValueError, match="modulation must be a non-empty string"):
        DecoderRequest(iq=iq, hypothesis=bad_cand_mod, sample_rate=100000.0)

    # Non-positive symbol rate
    bad_cand_sym = HypothesisCandidate(id="test", modulation="BPSK", symbolRate=0.0)
    with pytest.raises(ValueError, match="symbolRate must be strictly positive"):
        DecoderRequest(iq=iq, hypothesis=bad_cand_sym, sample_rate=100000.0)


# =====================================================================
# 2. Stage Result Contracts Tests
# =====================================================================

def test_synchronization_result_states():
    """Tests SynchronizationResult for success, failed, not_supported, not_evaluated."""
    # Success
    s_success = SynchronizationResult.create_success(
        synchronized_signal=np.ones((2, 100)),
        carrier_frequency_offset=125.5,
        timing_score=0.92,
        metrics={"snr_post_sync": 18.5},
    )
    assert s_success.success is True
    assert s_success.status == DecoderStageStatus.SUCCESS
    assert s_success.carrier_frequency_offset == 125.5
    assert s_success.timing_score == 0.92
    d_succ = s_success.to_dict()
    assert d_succ["status"] == "success"
    assert d_succ["has_synchronized_signal"] is True

    # Failed
    s_failed = SynchronizationResult.failed(
        failure_reason="Carrier PLL lost lock",
        metrics={"cycles_slipped": 14},
    )
    assert s_failed.success is False
    assert s_failed.status == DecoderStageStatus.FAILED
    assert s_failed.failure_reason == "Carrier PLL lost lock"

    # Not supported
    s_notsup = SynchronizationResult.not_supported(failure_reason="Continuous carrier sync not supported for burst signal")
    assert s_notsup.success is False
    assert s_notsup.status == DecoderStageStatus.NOT_SUPPORTED

    # Not evaluated
    s_noteval = SynchronizationResult.not_evaluated()
    assert s_noteval.success is False
    assert s_noteval.status == DecoderStageStatus.NOT_EVALUATED


def test_demodulation_result_states():
    """Tests DemodulationResult for all stage statuses."""
    symbols = np.array([1+1j, -1+1j, -1-1j, 1-1j])
    bits = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.uint8)

    # Success
    d_succ = DemodulationResult.create_success(
        symbols=symbols,
        bits=bits,
        constellation_score=0.88,
        quality_metrics={"evm_db": -22.4},
    )
    assert d_succ.success is True
    assert d_succ.status == DecoderStageStatus.SUCCESS
    assert d_succ.constellation_score == 0.88
    d_dict = d_succ.to_dict()
    assert d_dict["num_symbols"] == 4
    assert d_dict["num_bits"] == 8

    # Failed
    d_fail = DemodulationResult.failed("Constellation cluster ambiguity")
    assert d_fail.success is False
    assert d_fail.status == DecoderStageStatus.FAILED

    # Not supported
    d_notsup = DemodulationResult.not_supported("Modulation 256QAM not supported in this decoder")
    assert d_notsup.status == DecoderStageStatus.NOT_SUPPORTED

    # Not evaluated
    d_noteval = DemodulationResult.not_evaluated()
    assert d_noteval.status == DecoderStageStatus.NOT_EVALUATED


def test_interleaver_result_states():
    """Tests InterleaverResult for all stage statuses."""
    bits = np.array([0, 1, 0, 1, 1, 0], dtype=np.uint8)

    i_succ = InterleaverResult.create_success(deinterleaved_bits=bits, metrics={"block_size": 6})
    assert i_succ.success is True
    assert i_succ.attempted is True
    assert i_succ.status == DecoderStageStatus.SUCCESS

    i_fail = InterleaverResult.failed("Bitstream length not a multiple of interleaver matrix width")
    assert i_fail.success is False
    assert i_fail.status == DecoderStageStatus.FAILED

    i_notsup = InterleaverResult.not_supported("Interleaver block_64x64 not supported")
    assert i_notsup.status == DecoderStageStatus.NOT_SUPPORTED
    assert i_notsup.supported is False

    i_noteval = InterleaverResult.not_evaluated()
    assert i_noteval.status == DecoderStageStatus.NOT_EVALUATED
    assert i_noteval.attempted is False


def test_fec_result_states():
    """Tests FECResult for all stage statuses."""
    bits = np.array([1, 0, 1, 1, 0], dtype=np.uint8)

    # Success
    f_succ = FECResult.create_success(
        decoded_bits=bits,
        corrected_bits_count=3,
        error_rate=0.012,
        fec_score=0.95,
    )
    assert f_succ.success is True
    assert f_succ.status == DecoderStageStatus.SUCCESS
    assert f_succ.corrected_bits_count == 3
    assert f_succ.uncorrectable_errors is False
    d = f_succ.to_dict()
    assert d["num_decoded_bits"] == 5
    assert d["corrected_bits_count"] == 3

    # Failed
    f_fail = FECResult.failed("Viterbi metric divergence / uncorrectable errors")
    assert f_fail.success is False
    assert f_fail.status == DecoderStageStatus.FAILED
    assert f_fail.uncorrectable_errors is True

    # Not supported
    f_notsup = FECResult.not_supported("LDPC code rate 5/6 not supported")
    assert f_notsup.status == DecoderStageStatus.NOT_SUPPORTED
    assert f_notsup.supported is False

    # Not evaluated
    f_noteval = FECResult.not_evaluated()
    assert f_noteval.status == DecoderStageStatus.NOT_EVALUATED


def test_validation_result_states():
    """Tests ValidationResult for all stage statuses."""
    # Success
    v_succ = ValidationResult.create_success(
        is_valid=True,
        crc_status="valid",
        validation_score=1.0,
        metrics={"crc_poly": "CRC-16-CCITT"},
    )
    assert v_succ.success is True
    assert v_succ.status == DecoderStageStatus.SUCCESS
    assert v_succ.crc_status == "valid"

    # Failed
    v_fail = ValidationResult.failed(
        failure_reason="CRC checksum mismatch",
        crc_status="invalid",
    )
    assert v_fail.success is False
    assert v_fail.status == DecoderStageStatus.FAILED
    assert v_fail.crc_status == "invalid"

    # Not supported
    v_notsup = ValidationResult.not_supported("Custom 32-bit CRC poly not supported")
    assert v_notsup.status == DecoderStageStatus.NOT_SUPPORTED

    # Not evaluated
    v_noteval = ValidationResult.not_evaluated()
    assert v_noteval.status == DecoderStageStatus.NOT_EVALUATED
    assert v_noteval.crc_status == "not_evaluated"


# =====================================================================
# 3. DecoderResult & Serialization Tests
# =====================================================================

def test_decoder_result_successful_construction():
    """Tests a comprehensive successful DecoderResult."""
    bits = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
    symbols = np.array([1+0j, 0+1j, -1+0j, 0-1j])

    res = DecoderResult(
        success=True,
        hypothesis_id="hyp_qpsk_01",
        bits=bits,
        symbols=symbols,
        synchronization=SynchronizationResult.create_success(timing_score=0.9),
        demodulation=DemodulationResult.create_success(symbols=symbols, constellation_score=0.85),
        fec=FECResult.create_success(decoded_bits=bits, fec_score=0.95),
        validation=ValidationResult.create_success(is_valid=True, validation_score=1.0),
    )

    assert res.success is True
    assert res.bits is not None
    assert len(res.bits) == 8

    # Serialization test
    res_dict = res.to_dict()
    assert res_dict["success"] is True
    assert res_dict["hypothesis_id"] == "hyp_qpsk_01"
    assert res_dict["has_bits"] is True
    assert res_dict["bits_count"] == 8
    assert res_dict["synchronization"]["status"] == "success"
    assert res_dict["demodulation"]["status"] == "success"
    assert res_dict["fec"]["status"] == "success"
    assert res_dict["validation"]["status"] == "success"

    # Verify JSON serializability
    json_str = json.dumps(res_dict)
    assert isinstance(json_str, str)
    assert "hyp_qpsk_01" in json_str


def test_decoder_result_failed_construction():
    """Tests a failed DecoderResult with explicit failure reasons."""
    res = DecoderResult(
        success=False,
        hypothesis_id="hyp_qpsk_02",
        synchronization=SynchronizationResult.failed("Symbol timing PLL could not lock"),
        demodulation=DemodulationResult.not_evaluated(),
        fec=FECResult.not_evaluated(),
        validation=ValidationResult.not_evaluated(),
        failure_reason="Synchronization failure at carrier acquisition stage",
    )

    assert res.success is False
    assert res.bits is None
    assert res.failure_reason == "Synchronization failure at carrier acquisition stage"
    assert res.synchronization.status == DecoderStageStatus.FAILED
    assert res.demodulation.status == DecoderStageStatus.NOT_EVALUATED

    res_dict = res.to_dict()
    assert res_dict["success"] is False
    assert res_dict["has_bits"] is False
    assert res_dict["bits_count"] == 0
    assert res_dict["synchronization"]["status"] == "failed"
    assert res_dict["demodulation"]["status"] == "not_evaluated"


def test_decoder_result_unavailable_optional_fields():
    """Verifies that missing or unrun stages do not fabricate arbitrary numbers."""
    res = DecoderResult(
        success=False,
        hypothesis_id="hyp_unrun",
    )

    # Defaults must be NOT_EVALUATED
    assert res.synchronization.status == DecoderStageStatus.NOT_EVALUATED
    assert res.demodulation.status == DecoderStageStatus.NOT_EVALUATED
    assert res.interleaver.status == DecoderStageStatus.NOT_EVALUATED
    assert res.fec.status == DecoderStageStatus.NOT_EVALUATED
    assert res.validation.status == DecoderStageStatus.NOT_EVALUATED
    assert res.bits is None
    assert res.symbols is None

    # Serialization should not inject fabricated scores
    d = res.to_dict()
    assert "timing_score" not in d["synchronization"]
    assert "constellation_score" not in d["demodulation"]
    assert "fec_score" not in d["fec"]
    assert "validation_score" not in d["validation"]


# =====================================================================
# 4. DecoderContext Tests
# =====================================================================

def test_decoder_context_creation_and_mutation():
    """Verifies that DecoderContext initializes cleanly from request and accepts stage outputs."""
    iq = np.ones((2, 500), dtype=np.float32)
    cand = _make_candidate("8PSK", 20000.0)
    req = DecoderRequest(iq=iq, hypothesis=cand, sample_rate=200000.0, context={"channel_id": 4})

    ctx = DecoderContext.from_request(req)
    assert ctx.sample_rate == 200000.0
    assert ctx.modulation == "8PSK"
    assert ctx.symbol_rate == 20000.0
    assert ctx.metadata["channel_id"] == 4

    # Pipeline stages mutate context cleanly
    ctx.synchronized_iq = iq
    ctx.sync_result = SynchronizationResult.create_success(timing_score=0.95)
    ctx.symbols = np.ones(100, dtype=np.complex64)
    ctx.raw_bits = np.array([0, 1, 0], dtype=np.uint8)

    assert ctx.synchronized_iq is not None
    assert ctx.sync_result.success is True
    assert len(ctx.symbols) == 100
    assert len(ctx.raw_bits) == 3


# =====================================================================
# 5. DecoderEngine & ScaffoldDecoderEngine Tests
# =====================================================================

def test_scaffold_decoder_engine_non_fabrication():
    """
    Verifies that ScaffoldDecoderEngine implements DecoderEngine,
    validates the request, and returns NOT_EVALUATED without fake success.
    """
    engine = ScaffoldDecoderEngine()
    assert isinstance(engine, DecoderEngine)

    iq = np.ones((2, 200), dtype=np.float32)
    cand = _make_candidate("QPSK", 5000.0)
    req = DecoderRequest(iq=iq, hypothesis=cand, sample_rate=50000.0)

    result = engine.decode(req)

    # Must NOT fabricate success
    assert result.success is False
    assert result.hypothesis_id == cand.id
    assert result.synchronization.status == DecoderStageStatus.NOT_EVALUATED
    assert result.demodulation.status == DecoderStageStatus.NOT_EVALUATED
    assert result.interleaver.status == DecoderStageStatus.NOT_EVALUATED
    assert result.fec.status == DecoderStageStatus.NOT_EVALUATED
    assert result.validation.status == DecoderStageStatus.NOT_EVALUATED
    assert result.failure_reason is not None
    assert "Scaffold decoder engine" in result.failure_reason


# =====================================================================
# 6. Contract Integration with Existing Hypothesis & Scoring
# =====================================================================

def test_decoder_result_to_evidence_trace_mapping():
    """
    Verifies mapping of DecoderResult into EvidenceTrace:
    - sync -> timing
    - demod -> constellation
    - fec -> fec
    - validation -> bitstream
    Preserves ML and SNR from base trace.
    """
    base_trace = EvidenceTrace(
        ml=create_ml_evidence("BPSK", {"BPSK": 0.88}),
        symbol_rate=create_symbol_rate_evidence(1000.0, 1000.0),
    )

    dec_res = DecoderResult(
        success=True,
        hypothesis_id="hyp_test",
        synchronization=SynchronizationResult.create_success(timing_score=0.92),
        demodulation=DemodulationResult.create_success(constellation_score=0.85),
        fec=FECResult.create_success(fec_score=0.96),
        validation=ValidationResult.create_success(validation_score=1.0),
    )

    trace = decoder_result_to_evidence_trace(dec_res, base_trace)

    # Base evidence preserved
    assert trace.ml.status == EvidenceStatus.AVAILABLE
    assert trace.ml.score == 0.88
    assert trace.symbol_rate.status == EvidenceStatus.AVAILABLE

    # Decoder evidence updated to AVAILABLE with correct scores
    assert trace.timing.status == EvidenceStatus.AVAILABLE
    assert trace.timing.score == 0.92

    assert trace.constellation.status == EvidenceStatus.AVAILABLE
    assert trace.constellation.score == 0.85

    assert trace.fec.status == EvidenceStatus.AVAILABLE
    assert trace.fec.score == 0.96

    assert trace.bitstream.status == EvidenceStatus.AVAILABLE
    assert trace.bitstream.score == 1.0


def test_decoder_result_failed_stages_map_to_failed_evidence():
    """Verifies that FAILED decoder stages map to EvidenceStatus.FAILED with score 0.0."""
    dec_res = DecoderResult(
        success=False,
        hypothesis_id="hyp_fail",
        synchronization=SynchronizationResult.failed("Timing loss"),
        demodulation=DemodulationResult.not_supported("Modulation unsupported"),
        fec=FECResult.not_evaluated(),
        validation=ValidationResult.not_evaluated(),
    )

    trace = decoder_result_to_evidence_trace(dec_res)

    assert trace.timing.status == EvidenceStatus.FAILED
    assert trace.timing.score == 0.0
    assert trace.constellation.status == EvidenceStatus.NOT_SUPPORTED
    assert trace.fec.status == EvidenceStatus.NOT_EVALUATED
    assert trace.bitstream.status == EvidenceStatus.NOT_EVALUATED


def test_update_hypothesis_and_downstream_ranking():
    """
    Complete end-to-end integration test:
    1. Candidate hypothesis created.
    2. DecoderResult attached via update_hypothesis_from_decoder_result.
    3. calculate_raw_score evaluates evidence accurately.
    4. rank_hypotheses produces valid, normalized confidence scores.
    """
    cand1 = _make_candidate("BPSK", 10000.0, "cand_1")
    cand2 = _make_candidate("QPSK", 10000.0, "cand_2")

    # cand1 gets strong successful decoding result
    res1 = DecoderResult(
        success=True,
        hypothesis_id=cand1.id,
        synchronization=SynchronizationResult.create_success(timing_score=0.95),
        demodulation=DemodulationResult.create_success(constellation_score=0.90),
        fec=FECResult.create_success(fec_score=1.0),
        validation=ValidationResult.create_success(is_valid=True, validation_score=1.0),
    )
    update_hypothesis_from_decoder_result(cand1, res1)
    assert cand1.status == "success"

    # cand2 encounters decoding failure
    res2 = DecoderResult(
        success=False,
        hypothesis_id=cand2.id,
        synchronization=SynchronizationResult.failed("Timing error"),
        demodulation=DemodulationResult.failed("Demodulation error"),
        fec=FECResult.not_evaluated(),
        validation=ValidationResult.not_evaluated(),
        failure_reason="Carrier sync failed",
    )
    update_hypothesis_from_decoder_result(cand2, res2)
    assert cand2.status == "failed"

    # Run downstream scoring
    calculate_raw_score(cand1)
    calculate_raw_score(cand2)

    # cand1 must outscore cand2 due to positive decoder evidence
    assert cand1.rawScore > cand2.rawScore

    # Run ranking
    ranked = rank_hypotheses([cand1, cand2])
    assert len(ranked) == 2
    assert ranked[0].id == cand1.id
    assert ranked[0].confidenceScore > ranked[1].confidenceScore
    assert math.isclose(ranked[0].confidenceScore + ranked[1].confidenceScore, 1.0, rel_tol=1e-5)
