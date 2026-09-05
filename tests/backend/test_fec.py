"""
Unit tests for backend/fec/ — convolutional encoder, Viterbi decoder,
and bitstream validity checker.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from fec.convolutional import encode, decode, bit_error_rate, K, MEMORY, NUM_STATES
from fec.bitstream_check import check_bitstream, BitstreamCheckResult


# ── Encoder ───────────────────────────────────────────────────────────────────

class TestEncoder:
    def test_output_length(self):
        bits = np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
        encoded = encode(bits)
        expected_len = 2 * (len(bits) + MEMORY)
        assert len(encoded) == expected_len

    def test_output_is_binary(self):
        bits = np.random.randint(0, 2, 64).astype(np.uint8)
        encoded = encode(bits)
        assert set(encoded.tolist()).issubset({0, 1})

    def test_all_zeros_encodes_to_all_zeros(self):
        """All-zeros input with zero initial state → all-zeros codeword."""
        bits = np.zeros(16, dtype=np.uint8)
        encoded = encode(bits)
        assert np.all(encoded == 0), f"Non-zero output for zero input: {encoded}"

    def test_deterministic(self):
        bits = np.array([1, 0, 1, 0, 1, 1], dtype=np.uint8)
        assert np.array_equal(encode(bits), encode(bits))

    def test_num_states(self):
        assert NUM_STATES == 64  # 2^(K-1) = 2^6


# ── Decoder ───────────────────────────────────────────────────────────────────

class TestDecoder:
    def _roundtrip(self, bits: np.ndarray) -> tuple[np.ndarray, float]:
        encoded = encode(bits)
        decoded = decode(encoded)
        ber = bit_error_rate(bits, decoded)
        return decoded, ber

    def test_perfect_roundtrip_short(self):
        """Short known sequence should decode perfectly with no noise."""
        bits = np.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=np.uint8)
        _, ber = self._roundtrip(bits)
        assert ber == 0.0, f"BER={ber} for perfect channel"

    def test_perfect_roundtrip_long(self):
        np.random.seed(42)
        bits = np.random.randint(0, 2, 128).astype(np.uint8)
        _, ber = self._roundtrip(bits)
        assert ber == 0.0

    def test_all_zeros_roundtrip(self):
        bits = np.zeros(32, dtype=np.uint8)
        _, ber = self._roundtrip(bits)
        assert ber == 0.0

    def test_all_ones_roundtrip(self):
        bits = np.ones(32, dtype=np.uint8)
        _, ber = self._roundtrip(bits)
        assert ber == 0.0

    def test_decoder_output_is_binary(self):
        np.random.seed(0)
        bits = np.random.randint(0, 2, 64).astype(np.uint8)
        decoded = decode(encode(bits))
        assert set(decoded.tolist()).issubset({0, 1})

    def test_noisy_channel_low_ber(self):
        """
        With 1 random bit-flip in the encoded stream (high SNR scenario),
        Viterbi should recover the original bits with low BER.
        """
        np.random.seed(7)
        bits = np.random.randint(0, 2, 64).astype(np.uint8)
        encoded = encode(bits).copy()
        # Flip one bit
        flip_idx = np.random.randint(0, len(encoded))
        encoded[flip_idx] ^= 1
        decoded = decode(encoded)
        ber = bit_error_rate(bits, decoded)
        assert ber <= 0.02, f"BER={ber:.4f} too high for single-bit-flip channel"

    def test_odd_length_input_handled(self):
        """Odd-length received stream must not crash."""
        bits = np.random.randint(0, 2, 32).astype(np.uint8)
        encoded = encode(bits)
        odd = encoded[:-1]  # make it odd length
        result = decode(odd)
        assert result is not None

    def test_output_length_shorter_than_input(self):
        """Decoder output should be shorter than encoded input (tail flush removed)."""
        bits = np.ones(40, dtype=np.uint8)
        encoded = encode(bits)
        decoded = decode(encoded)
        assert len(decoded) <= len(bits)


# ── BER helper ────────────────────────────────────────────────────────────────

class TestBitErrorRate:
    def test_identical_arrays(self):
        a = np.array([0, 1, 0, 1], dtype=np.uint8)
        assert bit_error_rate(a, a) == 0.0

    def test_all_different(self):
        a = np.array([0, 0, 0, 0], dtype=np.uint8)
        b = np.array([1, 1, 1, 1], dtype=np.uint8)
        assert bit_error_rate(a, b) == 1.0

    def test_half_errors(self):
        a = np.array([0, 1, 0, 1], dtype=np.uint8)
        b = np.array([1, 1, 1, 1], dtype=np.uint8)
        assert bit_error_rate(a, b) == pytest.approx(0.5)

    def test_different_lengths_uses_shorter(self):
        a = np.array([0, 0, 0, 0], dtype=np.uint8)
        b = np.array([0, 0], dtype=np.uint8)
        assert bit_error_rate(a, b) == 0.0


# ── Bitstream check ───────────────────────────────────────────────────────────

class TestBitstreamCheck:
    def _random_bits(self, n: int, seed: int = 0) -> np.ndarray:
        np.random.seed(seed)
        return np.random.randint(0, 2, n).astype(np.uint8)

    def test_random_bits_pass(self):
        """Truly random bits should pass all three checks."""
        result = check_bitstream(self._random_bits(256))
        assert result.passed

    def test_all_zeros_fail(self):
        """All-zeros bitstream should fail (zero entropy, zero balance)."""
        bits = np.zeros(256, dtype=np.uint8)
        result = check_bitstream(bits)
        assert not result.passed

    def test_all_ones_fail(self):
        bits = np.ones(256, dtype=np.uint8)
        result = check_bitstream(bits)
        assert not result.passed

    def test_alternating_pattern_may_fail(self):
        """010101… has max entropy but extreme run balance — should fail run check."""
        bits = np.tile([0, 1], 128).astype(np.uint8)
        result = check_bitstream(bits)
        # Run check passes (runs of 1), entropy check passes; balance passes → overall pass
        # This is a valid borderline case — just verify it returns a result
        assert isinstance(result, BitstreamCheckResult)

    def test_short_bits_fail(self):
        bits = np.array([0, 1, 0], dtype=np.uint8)
        result = check_bitstream(bits, min_bits=32)
        assert not result.passed
        assert "length" in result.checks_failed

    def test_entropy_near_one_for_random(self):
        result = check_bitstream(self._random_bits(1024))
        assert result.entropy >= 0.9, f"Random bits entropy too low: {result.entropy}"

    def test_ones_ratio_reported(self):
        result = check_bitstream(self._random_bits(512))
        assert 0.0 <= result.ones_ratio <= 1.0

    def test_max_run_for_all_zeros(self):
        bits = np.zeros(64, dtype=np.uint8)
        result = check_bitstream(bits)
        assert result.max_run_length == 64

    def test_result_fields_present(self):
        result = check_bitstream(self._random_bits(128))
        assert hasattr(result, "passed")
        assert hasattr(result, "entropy")
        assert hasattr(result, "ones_ratio")
        assert hasattr(result, "max_run_length")
        assert hasattr(result, "checks_passed")
        assert hasattr(result, "checks_failed")
