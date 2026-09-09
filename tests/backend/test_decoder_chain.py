"""
Unit and integration tests for P5.3 Decoder-Chain Orchestration.
Verifies all 19 requirements specified in the P5.3 objective:
  1. Successful synchronization adapter execution with real/synthetic inputs.
  2. Unsupported synchronization configuration handling (NOT_SUPPORTED).
  3. Synchronization failure propagation (downstream stages NOT_EVALUATED).
  4. Supported demodulation mapping (BPSK, QPSK).
  5. Unsupported modulation handling (NOT_SUPPORTED).
  6. Demodulation failure propagation (downstream stages NOT_EVALUATED).
  7. Interleaver None handling (NOT_EVALUATED).
  8. Unsupported interleaver handling (NOT_SUPPORTED).
  9. FEC None handling (NOT_EVALUATED).
  10. Unsupported FEC handling (NOT_SUPPORTED).
  11. Validation unavailable handling (NOT_SUPPORTED, is_valid=False).
  12. Correct DecoderResult construction and diagnostic telemetry.
  13. DecoderResult.success remains False for partial decode.
  14. Stage statuses are exact and distinct.
  15. Intermediate context data is preserved in DecoderContext.
  16. Non-fabrication of evidence.
  17. Deterministic repeated execution.
  18. Custom / mock stage handler injection.
  19. Full end-to-end integration and hypothesis scoring compatibility.
"""
import math
import numpy as np
import pytest

from backend.hypothesis.candidates import (
    HypothesisCandidate,
    EvidenceStatus,
    EvidenceTrace,
)
from backend.hypothesis.evaluator import calculate_raw_score
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
    DecoderStage,
    SynchronizationAdapter,
    DemodulationAdapter,
    InterleaverAdapter,
    FECAdapter,
    ValidationAdapter,
    DecoderChain,
)


def _make_test_request(
    modulation: str = "BPSK",
    symbol_rate: float = 1000.0,
    fec_config: str = None,
    interleaver_config: str = None,
    sync_assumptions: dict = None,
    sample_rate: float = 10000.0,
    num_samples: int = 100,
) -> DecoderRequest:
    """Helper to construct a valid canonical DecoderRequest for testing."""
    # Canonical 2D real float32 array [2, N]
    t = np.arange(num_samples, dtype=np.float32)
    i = np.cos(2.0 * np.pi * 0.1 * t).astype(np.float32)
    q = np.sin(2.0 * np.pi * 0.1 * t).astype(np.float32)
    iq = np.vstack([i, q])

    cand = create_candidate(
        modulation=modulation,
        symbol_rate=symbol_rate,
        fec_config=fec_config,
        interleaver_config=interleaver_config,
        sync_assumptions=sync_assumptions,
    )
    return DecoderRequest(
        iq=iq,
        hypothesis=cand,
        sample_rate=sample_rate,
    )


# =====================================================================
# 1. Synchronization Tests
# =====================================================================

def test_synchronization_adapter_success_rrc():
    """Verifies that SynchronizationAdapter executes RRC matched filtering successfully."""
    req = _make_test_request(
        sync_assumptions={"matched_filter": "rrc", "filter_rolloff": 0.35}
    )
    ctx = DecoderContext.from_request(req)
    adapter = SynchronizationAdapter()

    res = adapter.execute(ctx)

    assert res.success is True
    assert res.status == DecoderStageStatus.SUCCESS
    assert ctx.synchronized_iq is not None
    assert ctx.synchronized_iq.shape == ctx.original_iq.shape
    assert res.timing_score is not None
    assert res.timing_score > 0.0


def test_synchronization_adapter_bypass_identity():
    """Verifies that SynchronizationAdapter handles bypass/none as identity pass-through."""
    req = _make_test_request(sync_assumptions={"matched_filter": "none"})
    ctx = DecoderContext.from_request(req)
    adapter = SynchronizationAdapter()

    res = adapter.execute(ctx)

    assert res.success is True
    assert res.status == DecoderStageStatus.SUCCESS
    assert np.allclose(ctx.synchronized_iq, ctx.original_iq)
    assert res.timing_score == 1.0


def test_synchronization_adapter_unsupported_carrier_recovery():
    """Verifies that requesting an unimplemented carrier recovery method returns NOT_SUPPORTED."""
    req = _make_test_request(
        sync_assumptions={"carrier_recovery": "costas", "matched_filter": "rrc"}
    )
    ctx = DecoderContext.from_request(req)
    adapter = SynchronizationAdapter()

    res = adapter.execute(ctx)

    assert res.success is False
    assert res.status == DecoderStageStatus.NOT_SUPPORTED
    assert "costas" in res.failure_reason.lower()


def test_synchronization_adapter_unsupported_timing_recovery():
    """Verifies that requesting an unimplemented timing recovery method returns NOT_SUPPORTED."""
    req = _make_test_request(
        sync_assumptions={"timing_recovery": "gardner", "matched_filter": "rrc"}
    )
    ctx = DecoderContext.from_request(req)
    adapter = SynchronizationAdapter()

    res = adapter.execute(ctx)

    assert res.success is False
    assert res.status == DecoderStageStatus.NOT_SUPPORTED
    assert "gardner" in res.failure_reason.lower()


def test_synchronization_failure_propagation():
    """
    Verifies that when synchronization fails, downstream stages (demodulation,
    interleaver, FEC, validation) are aborted and marked NOT_EVALUATED.
    """
    # Create failing sync adapter
    def fail_sync(ctx):
        return SynchronizationResult.failed(failure_reason="Carrier acquisition loss")

    chain = DecoderChain(sync_stage=SynchronizationAdapter(handler=fail_sync))
    req = _make_test_request(modulation="BPSK")

    res = chain.decode(req)

    assert res.success is False
    assert res.synchronization.status == DecoderStageStatus.FAILED
    assert res.synchronization.failure_reason == "Carrier acquisition loss"

    # Downstream stages MUST be NOT_EVALUATED
    assert res.demodulation.status == DecoderStageStatus.NOT_EVALUATED
    assert res.interleaver.status == DecoderStageStatus.NOT_EVALUATED
    assert res.fec.status == DecoderStageStatus.NOT_EVALUATED
    assert res.validation.status == DecoderStageStatus.NOT_EVALUATED

    assert res.failure_reason == "Carrier acquisition loss"


# =====================================================================
# 2. Demodulation Tests
# =====================================================================

def test_supported_demodulation_bpsk():
    """Verifies real BPSK demodulation with known constellation symbols."""
    # Synthetic BPSK symbols: [1, -1, 1, 1, -1] -> bits [1, 0, 1, 1, 0]
    ideal_syms = np.array([1.0, -1.0, 1.0, 1.0, -1.0], dtype=np.complex64)
    req = _make_test_request(modulation="BPSK", num_samples=5)
    ctx = DecoderContext.from_request(req)
    ctx.synchronized_iq = ideal_syms

    adapter = DemodulationAdapter()
    res = adapter.execute(ctx)

    assert res.success is True
    assert res.status == DecoderStageStatus.SUCCESS
    assert np.array_equal(res.bits, np.array([1, 0, 1, 1, 0], dtype=np.uint8))
    assert res.constellation_score >= 0.99
    assert ctx.symbols is not None
    assert ctx.raw_bits is not None


def test_supported_demodulation_qpsk():
    """Verifies real QPSK demodulation with known Gray-coded constellation points."""
    # QPSK points: 00 -> 1+1j, 01 -> -1+1j, 10 -> 1-1j, 11 -> -1-1j (scaled by 1/sqrt(2))
    scale = 1.0 / np.sqrt(2.0)
    qpsk_syms = np.array([1+1j, -1+1j, 1-1j, -1-1j], dtype=np.complex64) * scale

    req = _make_test_request(modulation="QPSK", num_samples=4)
    ctx = DecoderContext.from_request(req)
    ctx.synchronized_iq = qpsk_syms

    adapter = DemodulationAdapter()
    res = adapter.execute(ctx)

    assert res.success is True
    assert res.status == DecoderStageStatus.SUCCESS
    # Expected bits: 0,0, 0,1, 1,0, 1,1
    expected_bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
    assert np.array_equal(res.bits, expected_bits)
    assert res.constellation_score >= 0.99


def test_unsupported_modulation_handling():
    """Verifies that requesting an unimplemented modulation (e.g. 16QAM, 8PSK, FSK) returns NOT_SUPPORTED."""
    req_qam = _make_test_request(modulation="16QAM")
    ctx_qam = DecoderContext.from_request(req_qam)
    adapter = DemodulationAdapter()

    res_qam = adapter.execute(ctx_qam)
    assert res_qam.status == DecoderStageStatus.NOT_SUPPORTED
    assert "16QAM" in res_qam.failure_reason

    req_8psk = _make_test_request(modulation="8PSK")
    ctx_8psk = DecoderContext.from_request(req_8psk)
    res_8psk = adapter.execute(ctx_8psk)
    assert res_8psk.status == DecoderStageStatus.NOT_SUPPORTED

    req_fsk = _make_test_request(modulation="CPFSK")
    ctx_fsk = DecoderContext.from_request(req_fsk)
    res_fsk = adapter.execute(ctx_fsk)
    assert res_fsk.status == DecoderStageStatus.NOT_SUPPORTED


def test_demodulation_failure_propagation():
    """
    Verifies that when demodulation fails, downstream stages (interleaver, FEC, validation)
    are marked NOT_EVALUATED and the chain aborts cleanly.
    """
    # Create failing demod adapter
    def fail_demod(ctx):
        return DemodulationResult.failed(failure_reason="Constellation centroid collapse")

    chain = DecoderChain(demod_stage=DemodulationAdapter(handler=fail_demod))
    req = _make_test_request(modulation="BPSK")

    res = chain.decode(req)

    assert res.success is False
    assert res.synchronization.status == DecoderStageStatus.SUCCESS
    assert res.demodulation.status == DecoderStageStatus.FAILED
    assert res.demodulation.failure_reason == "Constellation centroid collapse"

    # Downstream stages MUST be NOT_EVALUATED
    assert res.interleaver.status == DecoderStageStatus.NOT_EVALUATED
    assert res.fec.status == DecoderStageStatus.NOT_EVALUATED
    assert res.validation.status == DecoderStageStatus.NOT_EVALUATED
    assert res.failure_reason == "Constellation centroid collapse"


# =====================================================================
# 3. Interleaver, FEC, and Validation Adapter Tests
# =====================================================================

def test_interleaver_none_handling():
    """Verifies that interleaver=None returns NOT_EVALUATED and passes raw_bits through."""
    req = _make_test_request(interleaver_config=None)
    ctx = DecoderContext.from_request(req)
    ctx.raw_bits = np.array([1, 0, 1], dtype=np.uint8)

    adapter = InterleaverAdapter()
    res = adapter.execute(ctx)

    assert res.status == DecoderStageStatus.NOT_EVALUATED
    assert np.array_equal(ctx.deinterleaved_bits, ctx.raw_bits)


def test_interleaver_unsupported_handling():
    """Verifies that configuring an unsupported interleaver returns NOT_SUPPORTED."""
    req = _make_test_request(interleaver_config="custom_unsupported_intl")
    ctx = DecoderContext.from_request(req)

    adapter = InterleaverAdapter()
    res = adapter.execute(ctx)

    assert res.status == DecoderStageStatus.NOT_SUPPORTED
    assert "custom_unsupported_intl" in res.failure_reason


def test_fec_none_handling():
    """Verifies that fec_config=None returns NOT_EVALUATED and passes bits through."""
    req = _make_test_request(fec_config=None)
    ctx = DecoderContext.from_request(req)
    ctx.raw_bits = np.array([1, 1, 0, 0], dtype=np.uint8)

    adapter = FECAdapter()
    res = adapter.execute(ctx)

    assert res.status == DecoderStageStatus.NOT_EVALUATED
    assert np.array_equal(ctx.decoded_bits, ctx.raw_bits)


def test_fec_unsupported_handling():
    """Verifies that configuring an unsupported FEC scheme returns NOT_SUPPORTED."""
    req = _make_test_request(fec_config="ldpc_r3/4")
    ctx = DecoderContext.from_request(req)

    adapter = FECAdapter()
    res = adapter.execute(ctx)

    assert res.status == DecoderStageStatus.NOT_SUPPORTED
    assert "ldpc_r3/4" in res.failure_reason


def test_validation_unavailable_handling():
    """Verifies that ValidationAdapter returns NOT_SUPPORTED and is_valid=False when unconfigured."""
    req = _make_test_request()
    ctx = DecoderContext.from_request(req)

    adapter = ValidationAdapter()
    res = adapter.execute(ctx)

    assert res.status == DecoderStageStatus.NOT_SUPPORTED
    assert res.is_valid is False
    assert res.success is False


# =====================================================================
# 4. DecoderChain Orchestration & DecoderResult Tests
# =====================================================================

def test_decoder_chain_partial_success_does_not_claim_overall_success():
    """
    CRITICAL REQUIREMENT:
    Even when synchronization and demodulation succeed completely, DecoderResult.success
    MUST remain False because validation is NOT_SUPPORTED and not valid.
    A partial DSP success is NOT a successful decode.
    """
    chain = DecoderChain()
    req = _make_test_request(modulation="BPSK")

    res = chain.decode(req)

    # Sync and Demod ran and succeeded
    assert res.synchronization.status == DecoderStageStatus.SUCCESS
    assert res.demodulation.status == DecoderStageStatus.SUCCESS
    assert res.bits is not None

    # But validation is not supported
    assert res.validation.status == DecoderStageStatus.NOT_SUPPORTED
    assert res.validation.is_valid is False

    # Therefore overall success MUST be False
    assert res.success is False


def test_decoder_chain_custom_mock_full_success():
    """
    Verifies that when a validated result IS produced (simulated with mock validation stage),
    DecoderResult.success cleanly becomes True.
    """
    def mock_validation(ctx):
        return ValidationResult.create_success(is_valid=True, crc_status="valid", validation_score=1.0)

    chain = DecoderChain(validation_stage=ValidationAdapter(handler=mock_validation))
    req = _make_test_request(modulation="BPSK")

    res = chain.decode(req)

    assert res.synchronization.status == DecoderStageStatus.SUCCESS
    assert res.demodulation.status == DecoderStageStatus.SUCCESS
    assert res.validation.status == DecoderStageStatus.SUCCESS
    assert res.validation.is_valid is True

    # Full decode pipeline validated -> True
    assert res.success is True


def test_decoder_result_construction_and_diagnostics():
    """Verifies that DecoderResult includes all stage outputs and diagnostic telemetry."""
    chain = DecoderChain()
    req = _make_test_request(modulation="BPSK", symbol_rate=2400.0)

    res = chain.decode(req)

    assert res.hypothesis_id == req.hypothesis.id
    assert res.diagnostic_details["orchestrator"] == "DecoderChain"
    assert res.diagnostic_details["modulation"] == "BPSK"
    assert res.diagnostic_details["sample_rate"] == 10000.0
    assert res.diagnostic_details["symbol_rate"] == 2400.0

    # Serialization test
    d = res.to_dict()
    assert isinstance(d, dict)
    assert d["success"] is False
    assert d["synchronization"]["status"] == "success"
    assert d["demodulation"]["status"] == "success"


def test_decoder_context_progressive_data_preservation():
    """Verifies that DecoderContext progressively holds intermediate representations."""
    req = _make_test_request(modulation="BPSK")
    ctx = DecoderContext.from_request(req)

    # Step 1: sync
    sync_stage = SynchronizationAdapter()
    sync_stage.execute(ctx)
    assert ctx.synchronized_iq is not None

    # Step 2: demod
    demod_stage = DemodulationAdapter()
    demod_stage.execute(ctx)
    assert ctx.symbols is not None
    assert ctx.raw_bits is not None

    # Step 3: interleaver (None -> passthrough)
    intl_stage = InterleaverAdapter()
    intl_stage.execute(ctx)
    assert np.array_equal(ctx.deinterleaved_bits, ctx.raw_bits)

    # Step 4: fec (None -> passthrough)
    fec_stage = FECAdapter()
    fec_stage.execute(ctx)
    assert np.array_equal(ctx.decoded_bits, ctx.raw_bits)


def test_non_fabricated_evidence_mapping():
    """
    Verifies that mapping DecoderResult to EvidenceTrace preserves explicit
    NOT_EVALUATED and NOT_SUPPORTED statuses without fabricating scores.
    """
    chain = DecoderChain()
    req = _make_test_request(modulation="BPSK")

    res = chain.decode(req)
    trace = decoder_result_to_evidence_trace(res)

    # Timing and Constellation should be AVAILABLE with genuine scores
    assert trace.timing.status == EvidenceStatus.AVAILABLE
    assert trace.timing.score is not None

    assert trace.constellation.status == EvidenceStatus.AVAILABLE
    assert trace.constellation.score is not None

    # FEC was bypassed (None) -> NOT_EVALUATED
    assert trace.fec.status == EvidenceStatus.NOT_EVALUATED
    assert trace.fec.score is None

    # Validation was NOT_SUPPORTED -> NOT_SUPPORTED
    assert trace.bitstream.status == EvidenceStatus.NOT_SUPPORTED
    assert trace.bitstream.score is None


def test_deterministic_repeated_chain_execution():
    """Verifies that repeated execution with identical inputs produces identical outputs."""
    chain = DecoderChain()
    req = _make_test_request(modulation="QPSK")

    res1 = chain.decode(req)
    res2 = chain.decode(req)

    assert res1.success == res2.success
    assert np.array_equal(res1.bits, res2.bits)
    assert np.allclose(res1.symbols, res2.symbols)
    assert res1.synchronization.status == res2.synchronization.status
    assert res1.demodulation.status == res2.demodulation.status
    assert res1.failure_reason == res2.failure_reason


def test_end_to_end_hypothesis_scoring_integration():
    """
    Verifies that a hypothesis candidate updated by DecoderChain cleanly integrates
    with calculate_raw_score and rank_hypotheses.
    """
    chain = DecoderChain()
    req = _make_test_request(modulation="BPSK")

    res = chain.decode(req)
    cand = req.hypothesis
    update_hypothesis_from_decoder_result(cand, res)

    # Raw scoring calculates without errors
    raw_score = calculate_raw_score(cand)
    assert raw_score > 0.0

    # Ranking works
    ranked = rank_hypotheses([cand])
    assert len(ranked) == 1
    assert ranked[0].confidenceScore == 1.0


# =====================================================================
# 5. P5.4 End-to-End Pipeline & Failure Propagation Tests
# =====================================================================

def test_decoder_chain_full_pipeline_success():
    """
    Verifies a complete, successful end-to-end decode:
      IQ -> Sync (Identity) -> Demod (BPSK) -> Deinterleaver (Block) -> FEC (Viterbi) -> Validation (CRC-16)
      -> Validated Payload and DecoderResult.success == True.
    """
    from backend.validation.crc import append_crc_to_bits
    from backend.fec.convolutional import encode_convolutional_r12_k7
    from backend.interleaver.block import interleave_block

    # 1. Original information payload (16 bits)
    payload = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1], dtype=np.uint8)

    # 2. Add CRC-16-CCITT (16 bits) -> 32 total bits
    frame_with_crc = append_crc_to_bits(payload, crc_name="CRC-16-CCITT")
    assert len(frame_with_crc) == 32

    # 3. Convolutional encode (Rate 1/2, K=7 with 6 tail bits) -> 2 * (32 + 6) = 76 bits
    coded_bits = encode_convolutional_r12_k7(frame_with_crc, tail_bits=True)
    assert len(coded_bits) == 76

    # 4. Block interleave with 4x19 matrix (4 * 19 = 76)
    interleaved_bits = interleave_block(coded_bits, rows=4, cols=19)
    assert len(interleaved_bits) == 76

    # 5. Modulate to ideal BPSK symbols: 0 -> -1.0, 1 -> +1.0
    symbols = (2.0 * interleaved_bits.astype(np.float32) - 1.0).astype(np.complex64)

    # 6. Construct 2D real IQ array [2, N]
    iq = np.vstack([symbols.real, symbols.imag]).astype(np.float32)

    # 7. Create candidate and DecoderRequest
    cand = create_candidate(
        modulation="BPSK",
        symbol_rate=1000.0,
        fec_config="conv_r1/2_k7",
        interleaver_config="block_4x19",
    )
    req = DecoderRequest(
        iq=iq,
        hypothesis=cand,
        sample_rate=10000.0,
        context={"validation_config": {"crc_scheme": "CRC-16-CCITT"}},
    )

    # 8. Execute DecoderChain
    chain = DecoderChain()
    res = chain.decode(req)

    # 9. Verify every stage succeeded
    assert res.synchronization.status == DecoderStageStatus.SUCCESS
    assert res.demodulation.status == DecoderStageStatus.SUCCESS
    assert res.interleaver.status == DecoderStageStatus.SUCCESS
    assert res.fec.status == DecoderStageStatus.SUCCESS
    assert res.validation.status == DecoderStageStatus.SUCCESS
    assert res.validation.is_valid is True
    assert res.validation.crc_status == "valid"

    # Overall success MUST be True
    assert res.success is True
    assert res.failure_reason is None

    # Decoded bits contain the original payload with CRC
    assert np.array_equal(res.bits, frame_with_crc)


def test_decoder_chain_interleaver_failure_aborts_downstream():
    """
    Verifies that an interleaver failure (e.g. block size mismatch) aborts downstream
    stages (FEC and validation remain NOT_EVALUATED), and overall success is False.
    """
    # 50 samples is not a multiple of 16x16 = 256
    req = _make_test_request(modulation="BPSK", interleaver_config="block_16x16", num_samples=50)
    req.context = {"validation_config": {"crc_scheme": "CRC-16-CCITT"}}

    chain = DecoderChain()
    res = chain.decode(req)

    assert res.synchronization.status == DecoderStageStatus.SUCCESS
    assert res.demodulation.status == DecoderStageStatus.SUCCESS
    assert res.interleaver.status == DecoderStageStatus.FAILED
    assert res.fec.status == DecoderStageStatus.NOT_EVALUATED
    assert res.validation.status == DecoderStageStatus.NOT_EVALUATED
    assert res.success is False
    assert "not a multiple" in res.failure_reason


def test_decoder_chain_crc_mismatch_fails_overall_success():
    """
    Verifies that when sync, demod, interleaver, and FEC all succeed, but CRC checksum
    mismatches, validation reports FAILED and overall success is strictly False.
    """
    from backend.fec.convolutional import encode_convolutional_r12_k7

    # Construct bits with invalid CRC
    bad_frame = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
    coded_bits = encode_convolutional_r12_k7(bad_frame, tail_bits=True)
    symbols = (2.0 * coded_bits.astype(np.float32) - 1.0).astype(np.complex64)
    iq = np.vstack([symbols.real, symbols.imag]).astype(np.float32)

    cand = create_candidate(
        modulation="BPSK",
        symbol_rate=1000.0,
        fec_config="conv_r1/2_k7",
        interleaver_config=None,
    )
    req = DecoderRequest(
        iq=iq,
        hypothesis=cand,
        sample_rate=10000.0,
        context={"validation_config": {"crc_scheme": "CRC-16-CCITT"}},
    )

    chain = DecoderChain()
    res = chain.decode(req)

    assert res.synchronization.status == DecoderStageStatus.SUCCESS
    assert res.demodulation.status == DecoderStageStatus.SUCCESS
    assert res.interleaver.status == DecoderStageStatus.NOT_EVALUATED  # None -> bypass
    assert res.fec.status == DecoderStageStatus.SUCCESS
    assert res.validation.status == DecoderStageStatus.FAILED
    assert res.validation.is_valid is False
    assert res.validation.crc_status == "invalid"
    assert res.success is False
    assert "checksum mismatch" in res.failure_reason
