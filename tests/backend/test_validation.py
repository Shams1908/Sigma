"""
Tests for CRC and Frame Validation modules in SIGMA.
Validates:
  1. Standard known-answer test vectors for CRC-16-CCITT, CRC-32, and CRC-8.
  2. Bit-level CRC calculation and serialization.
  3. Frame validation on clean frames (payload + matching CRC).
  4. Detection of 1-bit and multi-bit corruptions.
  5. Sync-word (preamble) correlation and alignment.
  6. Rejection of truncated frames shorter than CRC width.
  7. ValidationAdapter execution and non-fabrication of scores.
"""
import pytest
import numpy as np

from backend.validation.crc import (
    compute_crc_bytes,
    compute_crc_bits,
    check_crc_bits,
    append_crc_to_bits,
    get_crc_param,
)
from backend.validation.frame_validator import FrameValidator
from backend.hypothesis.decoder_contracts import (
    ValidationResult,
    DecoderStageStatus,
    DecoderRequest,
    DecoderContext,
)
from backend.hypothesis.decoder_chain import ValidationAdapter
from backend.hypothesis.generator import create_candidate


# =====================================================================
# 1. Standard Known-Answer Vector Tests
# =====================================================================

def test_crc16_ccitt_standard_vector():
    """
    Standard check vector: ASCII '123456789'
    CRC-16-CCITT (poly 0x1021, init 0xFFFF, refIn=False, refOut=False, xorOut=0x0000)
    Expected result: 0x29B1
    """
    data = b"123456789"
    crc = compute_crc_bytes(data, crc_name="CRC-16-CCITT")
    assert crc == 0x29B1


def test_crc32_standard_vector():
    """
    Standard check vector: ASCII '123456789'
    CRC-32 (IEEE 802.3, poly 0x04C11DB7, init 0xFFFFFFFF, refIn=True, refOut=True, xorOut=0xFFFFFFFF)
    Expected result: 0xCBF43926
    """
    data = b"123456789"
    crc = compute_crc_bytes(data, crc_name="CRC-32")
    assert crc == 0xCBF43926


def test_crc8_standard_vector():
    """
    Standard check vector: ASCII '123456789'
    CRC-8 (poly 0x07, init 0x00, refIn=False, refOut=False, xorOut=0x00)
    Expected result: 0xF4
    """
    data = b"123456789"
    crc = compute_crc_bytes(data, crc_name="CRC-8")
    assert crc == 0xF4


# =====================================================================
# 2. Bit-Level CRC Append and Check Tests
# =====================================================================

def test_bit_level_crc16_roundtrip():
    """Verifies appending CRC-16 to payload and validating match."""
    payload = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0], dtype=np.uint8)
    frame = append_crc_to_bits(payload, crc_name="CRC-16-CCITT")

    # 16 payload bits + 16 CRC bits = 32 total bits
    assert len(frame) == 32

    is_valid, extracted_payload, expected_crc, observed_crc = check_crc_bits(frame, crc_name="CRC-16-CCITT")
    assert is_valid is True
    assert np.array_equal(extracted_payload, payload)
    assert expected_crc == observed_crc


def test_bit_level_crc16_detects_corruption():
    """Verifies that flipping even a single bit in the frame invalidates CRC."""
    payload = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0], dtype=np.uint8)
    frame = append_crc_to_bits(payload, crc_name="CRC-16-CCITT")

    # Flip 1 bit in the payload
    corrupted = frame.copy()
    corrupted[3] ^= 1

    is_valid, _, expected, observed = check_crc_bits(corrupted, crc_name="CRC-16-CCITT")
    assert is_valid is False
    assert expected != observed


def test_bit_level_crc32_roundtrip():
    """Verifies appending CRC-32 to arbitrary length payload."""
    payload = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
    frame = append_crc_to_bits(payload, crc_name="CRC-32")

    assert len(frame) == len(payload) + 32
    is_valid, extracted, expected, observed = check_crc_bits(frame, crc_name="CRC-32")
    assert is_valid is True
    assert np.array_equal(extracted, payload)
    assert expected == observed


# =====================================================================
# 3. FrameValidator with Sync Word Correlation
# =====================================================================

def test_frame_validator_valid_frame_without_sync():
    """Verifies FrameValidator on clean payload with CRC-16."""
    payload = np.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=np.uint8)
    frame = append_crc_to_bits(payload, crc_name="CRC-16-CCITT")

    validator = FrameValidator()
    res = validator.validate(bits=frame, crc_scheme="CRC-16-CCITT")

    assert res.status == DecoderStageStatus.SUCCESS
    assert res.is_valid is True
    assert res.crc_status == "valid"
    assert res.validation_score == 1.0
    assert res.metrics["payload_bits_count"] == 8


def test_frame_validator_detects_mismatch():
    """Verifies FrameValidator failure when checksum is mismatched."""
    payload = np.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=np.uint8)
    frame = append_crc_to_bits(payload, crc_name="CRC-16-CCITT")
    frame[0] ^= 1  # Corrupt

    validator = FrameValidator()
    res = validator.validate(bits=frame, crc_scheme="CRC-16-CCITT")

    assert res.status == DecoderStageStatus.FAILED
    assert res.is_valid is False
    assert res.crc_status == "invalid"
    assert "checksum mismatch" in res.failure_reason


def test_frame_validator_sync_word_detection():
    """Verifies alignment to sync word preamble embedded in noise bits."""
    sync_word = np.array([1, 1, 1, 0, 1, 1, 0, 0], dtype=np.uint8)  # 8-bit sync
    payload = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.uint8)
    frame_body = append_crc_to_bits(payload, crc_name="CRC-16-CCITT")

    # Prepend 10 arbitrary bits before sync word
    noise_prefix = np.array([0, 0, 1, 0, 1, 1, 0, 1, 0, 1], dtype=np.uint8)
    full_stream = np.concatenate([noise_prefix, sync_word, frame_body])

    validator = FrameValidator()
    # Pass sync word and expected frame length
    expected_len = len(sync_word) + len(frame_body)
    res = validator.validate(
        bits=full_stream,
        crc_scheme="CRC-16-CCITT",
        sync_word=sync_word,
    )

    # Working bits after sync word should contain frame_body
    # Here working_bits was sliced starting from sync word (len = sync_word + frame_body)
    assert res.frame_status == "aligned_at_bit_10"


def test_frame_validator_sync_word_not_found():
    """Verifies failure when requested sync word is absent."""
    bits = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8)
    missing_sync = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)

    validator = FrameValidator()
    res = validator.validate(bits=bits, crc_scheme="CRC-16-CCITT", sync_word=missing_sync)

    assert res.status == DecoderStageStatus.FAILED
    assert res.is_valid is False
    assert "not found" in res.failure_reason


def test_frame_validator_truncated_frame():
    """Verifies failure when bitstream is shorter than CRC width."""
    bits = np.array([1, 0, 1], dtype=np.uint8)  # 3 bits < 16 bits
    validator = FrameValidator()
    res = validator.validate(bits=bits, crc_scheme="CRC-16-CCITT")

    assert res.status == DecoderStageStatus.FAILED
    assert "too short" in res.failure_reason


# =====================================================================
# 4. ValidationAdapter Integration Tests
# =====================================================================

def test_validation_adapter_unsupported_when_unconfigured():
    """
    CRITICAL: Verifies ValidationAdapter returns NOT_SUPPORTED when no validation
    criteria or CRC scheme are specified for the candidate.
    """
    cand = create_candidate(modulation="BPSK", symbol_rate=1000.0)
    req = DecoderRequest(
        iq=np.ones((2, 100), dtype=np.float32),
        hypothesis=cand,
        sample_rate=10000.0,
    )
    ctx = DecoderContext.from_request(req)
    ctx.raw_bits = np.array([1, 0, 1, 1, 0, 0], dtype=np.uint8)

    adapter = ValidationAdapter()
    res = adapter.execute(ctx)

    assert res.status == DecoderStageStatus.NOT_SUPPORTED
    assert res.is_valid is False
    assert res.success is False


def test_validation_adapter_executes_with_metadata_crc():
    """Verifies ValidationAdapter executes FrameValidator when crc_scheme is in metadata."""
    payload = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    frame = append_crc_to_bits(payload, crc_name="CRC-16-CCITT")

    cand = create_candidate(modulation="BPSK", symbol_rate=1000.0)
    req = DecoderRequest(
        iq=np.ones((2, 100), dtype=np.float32),
        hypothesis=cand,
        sample_rate=10000.0,
        context={"validation_config": {"crc_scheme": "CRC-16-CCITT"}},
    )
    ctx = DecoderContext.from_request(req)
    ctx.raw_bits = frame

    adapter = ValidationAdapter()
    res = adapter.execute(ctx)

    assert res.status == DecoderStageStatus.SUCCESS
    assert res.is_valid is True
    assert res.crc_status == "valid"


def test_validation_adapter_rejects_unsupported_crc_scheme():
    """Verifies ValidationAdapter returns NOT_SUPPORTED for unknown CRC scheme."""
    cand = create_candidate(modulation="BPSK", symbol_rate=1000.0)
    req = DecoderRequest(
        iq=np.ones((2, 100), dtype=np.float32),
        hypothesis=cand,
        sample_rate=10000.0,
        context={"crc_scheme": "CRC-64-ECMA-UNSUPPORTED"},
    )
    ctx = DecoderContext.from_request(req)
    ctx.raw_bits = np.array([1, 0, 1, 0], dtype=np.uint8)

    adapter = ValidationAdapter()
    res = adapter.execute(ctx)

    assert res.status == DecoderStageStatus.NOT_SUPPORTED
    assert "not supported" in res.failure_reason
