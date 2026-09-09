"""
Tests for Interleaver and Deinterleaver modules in SIGMA.
Validates:
  1. Deterministic rectangular matrix block interleaving.
  2. Roundtrip deinterleaving exact bit recovery: deinterleave(interleave(x)) == x.
  3. Multiple consecutive blocks deinterleaving.
  4. Bitstream length validation (non-multiples of block size).
  5. Invalid configuration parsing (negative, zero, malformed string).
  6. InterleaverAdapter execution and status contracts.
"""
import pytest
import numpy as np

from backend.interleaver.block import (
    interleave_block,
    deinterleave_block,
    BlockDeinterleaver,
    parse_block_interleaver_config,
)
from backend.hypothesis.decoder_contracts import (
    InterleaverResult,
    DecoderStageStatus,
    DecoderRequest,
    DecoderContext,
)
from backend.hypothesis.decoder_chain import InterleaverAdapter
from backend.hypothesis.generator import create_candidate


# =====================================================================
# 1. Config Parsing Tests
# =====================================================================

def test_parse_block_interleaver_config():
    """Verifies parsing of standard block interleaver string configurations."""
    assert parse_block_interleaver_config("block_16x16") == (16, 16)
    assert parse_block_interleaver_config("block_4x8") == (4, 8)
    assert parse_block_interleaver_config("matrix_8x16") == (8, 16)
    assert parse_block_interleaver_config("BLOCK_8X8") == (8, 8)

    # Invalid configurations
    assert parse_block_interleaver_config(None) is None
    assert parse_block_interleaver_config("") is None
    assert parse_block_interleaver_config("none") is None
    assert parse_block_interleaver_config("block_0x16") is None
    assert parse_block_interleaver_config("random_interleaver") is None


# =====================================================================
# 2. Known-Answer Matrix Transposition Tests
# =====================================================================

def test_block_interleaver_known_answer_matrix():
    """
    Verifies that a 2x3 matrix is written row-by-row and read column-by-column:
      Row 0: [0, 1, 2]
      Row 1: [3, 4, 5]
      Read columns -> [0, 3, 1, 4, 2, 5]
    """
    bits = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint8)
    interleaved = interleave_block(bits, rows=2, cols=3)
    expected = np.array([0, 3, 1, 4, 2, 5], dtype=np.uint8)

    assert np.array_equal(interleaved, expected)

    # Deinterleaver restores original
    restored = deinterleave_block(interleaved, rows=2, cols=3)
    assert np.array_equal(restored, bits)


def test_block_interleaver_multi_block_roundtrip():
    """Verifies roundtrip preservation across multiple consecutive blocks."""
    # 4 blocks of 4x4 (16 bits/block = 64 total bits)
    rng = np.random.default_rng(42)
    original_bits = rng.integers(0, 2, size=64, dtype=np.uint8)

    interleaved = interleave_block(original_bits, rows=4, cols=4)
    # Ensure permutation actually shuffled bits
    assert not np.array_equal(interleaved, original_bits)

    restored = deinterleave_block(interleaved, rows=4, cols=4)
    assert np.array_equal(restored, original_bits)


def test_block_16x16_standard_roundtrip():
    """Verifies standard block_16x16 (256 bits per block) execution."""
    rng = np.random.default_rng(123)
    original_bits = rng.integers(0, 2, size=512, dtype=np.uint8)  # Exactly 2 blocks of 256

    interleaved = interleave_block(original_bits, rows=16, cols=16)
    assert len(interleaved) == 512

    restored = deinterleave_block(interleaved, rows=16, cols=16)
    assert np.array_equal(restored, original_bits)


# =====================================================================
# 3. Input Validation & Error Handling
# =====================================================================

def test_block_deinterleaver_invalid_length():
    """Verifies rejection when bitstream length is not a multiple of block size."""
    bits = np.array([1, 0, 1, 1, 0], dtype=np.uint8)  # 5 bits, not multiple of 4x4=16
    deinterleaver = BlockDeinterleaver(rows=4, cols=4)
    res = deinterleaver.deinterleave(bits)

    assert res.status == DecoderStageStatus.FAILED
    assert res.success is False
    assert "not a multiple" in res.failure_reason
    assert res.metrics["block_size"] == 16


def test_block_deinterleaver_empty_bits():
    """Verifies handling of empty bit input."""
    deinterleaver = BlockDeinterleaver(rows=4, cols=4)
    res = deinterleaver.deinterleave(np.array([], dtype=np.uint8))

    assert res.status == DecoderStageStatus.FAILED
    assert "No input bits" in res.failure_reason


# =====================================================================
# 4. InterleaverAdapter Integration Tests
# =====================================================================

def test_interleaver_adapter_bypassed_when_none():
    """Verifies InterleaverAdapter returns NOT_EVALUATED and passes bits when config is None."""
    cand = create_candidate(modulation="BPSK", symbol_rate=1000.0, interleaver_config=None)
    req = DecoderRequest(
        iq=np.ones((2, 100), dtype=np.float32),
        hypothesis=cand,
        sample_rate=10000.0,
    )
    ctx = DecoderContext.from_request(req)
    ctx.raw_bits = np.array([1, 0, 1, 0], dtype=np.uint8)

    adapter = InterleaverAdapter()
    res = adapter.execute(ctx)

    assert res.status == DecoderStageStatus.NOT_EVALUATED
    assert np.array_equal(ctx.deinterleaved_bits, ctx.raw_bits)


def test_interleaver_adapter_executes_block_16x16():
    """Verifies InterleaverAdapter executes BlockDeinterleaver for block_16x16."""
    rng = np.random.default_rng(999)
    original_bits = rng.integers(0, 2, size=256, dtype=np.uint8)
    interleaved = interleave_block(original_bits, rows=16, cols=16)

    cand = create_candidate(modulation="BPSK", symbol_rate=1000.0, interleaver_config="block_16x16")
    req = DecoderRequest(
        iq=np.ones((2, 100), dtype=np.float32),
        hypothesis=cand,
        sample_rate=10000.0,
    )
    ctx = DecoderContext.from_request(req)
    ctx.raw_bits = interleaved

    adapter = InterleaverAdapter()
    res = adapter.execute(ctx)

    assert res.status == DecoderStageStatus.SUCCESS
    assert res.success is True
    assert np.array_equal(ctx.deinterleaved_bits, original_bits)
    assert res.metrics["block_size"] == 256


def test_interleaver_adapter_returns_not_supported_for_unknown():
    """Verifies InterleaverAdapter returns NOT_SUPPORTED for unrecognized configurations."""
    cand = create_candidate(modulation="BPSK", symbol_rate=1000.0, interleaver_config="convolutional_intl")
    req = DecoderRequest(
        iq=np.ones((2, 100), dtype=np.float32),
        hypothesis=cand,
        sample_rate=10000.0,
    )
    ctx = DecoderContext.from_request(req)
    ctx.raw_bits = np.array([1, 0, 1, 0], dtype=np.uint8)

    adapter = InterleaverAdapter()
    res = adapter.execute(ctx)

    assert res.status == DecoderStageStatus.NOT_SUPPORTED
    assert "convolutional_intl" in res.failure_reason
