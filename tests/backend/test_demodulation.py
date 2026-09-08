"""
Unit tests for backend/demodulation/psk_demod.py

Tests cover BPSK and QPSK demodulators with:
  - Clean synthetic symbols (low EVM expected)
  - Noisy symbols (EVM < 0.5 at SNR=10 dB)
  - Edge cases (empty input, single symbol)
  - Correct dispatcher routing
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from demodulation.psk_demod import (
    DemodResult,
    demod_bpsk,
    demod_qpsk,
    demodulate_psk,
    constellation_score,
    _rms_evm,
)
from demodulation import demodulate


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_bpsk_symbols(n: int = 128) -> np.ndarray:
    np.random.seed(10)
    bits = np.random.randint(0, 2, n)
    return (2 * bits - 1).astype(np.complex64)


def _noisy_bpsk_symbols(snr_db: float = 10.0, n: int = 128) -> np.ndarray:
    symbols = _clean_bpsk_symbols(n)
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_std = np.sqrt(1.0 / (2.0 * snr_lin))
    noise = (noise_std * np.random.randn(n) + 1j * noise_std * np.random.randn(n)).astype(np.complex64)
    return symbols + noise


_QPSK_NORM = 1.0 / np.sqrt(2.0)
_QPSK_CONST = [
    _QPSK_NORM + 1j * _QPSK_NORM,
    -_QPSK_NORM + 1j * _QPSK_NORM,
    -_QPSK_NORM - 1j * _QPSK_NORM,
    _QPSK_NORM - 1j * _QPSK_NORM,
]


def _clean_qpsk_symbols(n: int = 128) -> np.ndarray:
    np.random.seed(11)
    indices = np.random.randint(0, 4, n)
    return np.array([_QPSK_CONST[i] for i in indices], dtype=np.complex64)


def _noisy_qpsk_symbols(snr_db: float = 10.0, n: int = 128) -> np.ndarray:
    symbols = _clean_qpsk_symbols(n)
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_std = np.sqrt(1.0 / (2.0 * snr_lin))
    noise = (noise_std * np.random.randn(n) + 1j * noise_std * np.random.randn(n)).astype(np.complex64)
    return symbols + noise


# ── BPSK tests ────────────────────────────────────────────────────────────────

class TestBPSK:
    def test_result_type(self):
        r = demod_bpsk(_clean_bpsk_symbols())
        assert isinstance(r, DemodResult)
        assert r.modulation == "BPSK"

    def test_bits_shape(self):
        n = 64
        r = demod_bpsk(_clean_bpsk_symbols(n))
        assert len(r.bits) == n
        assert set(r.bits.tolist()).issubset({0, 1})

    def test_clean_evm_low(self):
        r = demod_bpsk(_clean_bpsk_symbols())
        assert r.evm_rms < 0.05, f"Clean BPSK EVM too high: {r.evm_rms}"

    def test_noisy_evm_below_threshold(self):
        """At SNR=10 dB BPSK EVM should still be below the 0.5 pass threshold."""
        r = demod_bpsk(_noisy_bpsk_symbols(snr_db=10.0))
        assert r.evm_rms < 0.5, f"Noisy BPSK EVM too high: {r.evm_rms}"

    def test_decisions_are_valid_bpsk_points(self):
        r = demod_bpsk(_clean_bpsk_symbols())
        valid = {complex(1.0, 0.0), complex(-1.0, 0.0)}
        for d in r.decisions:
            assert complex(d) in valid, f"Invalid BPSK decision: {d}"

    def test_single_symbol(self):
        r = demod_bpsk(np.array([0.9 + 0.1j], dtype=np.complex64))
        assert len(r.bits) == 1
        assert r.bits[0] == 1  # real > 0 → bit 1 (canonical: +1 → bit 1)

    def test_negative_symbol(self):
        r = demod_bpsk(np.array([-0.9 + 0.1j], dtype=np.complex64))
        assert r.bits[0] == 0  # real < 0 → bit 0 (canonical: -1 → bit 0)


# ── QPSK tests ────────────────────────────────────────────────────────────────

class TestQPSK:
    def test_result_type(self):
        r = demod_qpsk(_clean_qpsk_symbols())
        assert isinstance(r, DemodResult)
        assert r.modulation == "QPSK"

    def test_bits_shape(self):
        n = 64
        r = demod_qpsk(_clean_qpsk_symbols(n))
        assert len(r.bits) == 2 * n  # 2 bits per symbol

    def test_bits_binary(self):
        r = demod_qpsk(_clean_qpsk_symbols())
        assert set(r.bits.tolist()).issubset({0, 1})

    def test_clean_evm_low(self):
        r = demod_qpsk(_clean_qpsk_symbols())
        assert r.evm_rms < 0.05, f"Clean QPSK EVM too high: {r.evm_rms}"

    def test_noisy_evm_below_threshold(self):
        r = demod_qpsk(_noisy_qpsk_symbols(snr_db=10.0))
        assert r.evm_rms < 0.5

    def test_quadrant_q1(self):
        """I > 0, Q > 0 → canonical bits 00."""
        r = demod_qpsk(np.array([0.7 + 0.7j], dtype=np.complex64))
        assert list(r.bits) == [0, 0]

    def test_quadrant_q2(self):
        """I < 0, Q > 0 → canonical bits 01."""
        r = demod_qpsk(np.array([-0.7 + 0.7j], dtype=np.complex64))
        assert list(r.bits) == [0, 1]

    def test_quadrant_q3(self):
        """I < 0, Q < 0 → canonical bits 11."""
        r = demod_qpsk(np.array([-0.7 - 0.7j], dtype=np.complex64))
        assert list(r.bits) == [1, 1]

    def test_quadrant_q4(self):
        """I > 0, Q < 0 → canonical bits 10."""
        r = demod_qpsk(np.array([0.7 - 0.7j], dtype=np.complex64))
        assert list(r.bits) == [1, 0]


# ── Dispatcher ────────────────────────────────────────────────────────────────

class TestDispatcher:
    def test_bpsk_routing(self):
        r = demodulate_psk(_clean_bpsk_symbols(), "BPSK")
        assert r.modulation == "BPSK"

    def test_qpsk_routing(self):
        r = demodulate_psk(_clean_qpsk_symbols(), "QPSK")
        assert r.modulation == "QPSK"

    def test_unsupported_raises(self):
        with pytest.raises(ValueError):
            demodulate_psk(_clean_bpsk_symbols(), "FSK")

    def test_demodulate_dispatcher_bpsk(self):
        r = demodulate(_clean_bpsk_symbols(), "BPSK")
        assert r.modulation == "BPSK"

    def test_demodulate_dispatcher_qpsk(self):
        r = demodulate(_clean_qpsk_symbols(), "QPSK")
        assert r.modulation == "QPSK"

    def test_16qam_raises_not_implemented(self):
        from demodulation.qam_demod import QAMDemodNotImplemented
        with pytest.raises(QAMDemodNotImplemented):
            demodulate(_clean_bpsk_symbols(), "16QAM")


# ── EVM and score ─────────────────────────────────────────────────────────────

class TestEVMAndScore:
    def test_evm_zero_for_perfect(self):
        perfect = np.array([1.0 + 0j, -1.0 + 0j], dtype=np.complex64)
        decisions = perfect.copy()
        assert _rms_evm(perfect, decisions) == pytest.approx(0.0, abs=1e-6)

    def test_constellation_score_perfect(self):
        r = demod_bpsk(_clean_bpsk_symbols())
        score = constellation_score(r)
        assert 0.9 <= score <= 1.0

    def test_constellation_score_noisy(self):
        r = demod_bpsk(_noisy_bpsk_symbols(snr_db=5.0))
        score = constellation_score(r)
        assert 0.0 <= score <= 1.0


# ── Generator → Demodulator round-trip tests ──────────────────────────────────
#
# These tests use the ML generator modulators as the source of truth.
# They prove that the backend demodulator exactly inverts the canonical
# bit→symbol mapping defined in the ML generators.

class TestGeneratorRoundTrip:
    """
    Round-trip: ML generator produces symbols from bits, backend demodulator
    recovers the exact original bit sequence.
    """

    def test_bpsk_roundtrip_exact(self):
        """
        BPSK: bits → BPSKModulator symbols → demod_bpsk → same bits.
        Uses the ML BPSKModulator (canonical: bit0→-1, bit1→+1).
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

        from ml.generators.modulation.bpsk import BPSKModulator
        from ml.generators.config import GeneratorConfig

        rng = np.random.default_rng(0)
        n_symbols = 256
        original_bits = rng.integers(0, 2, size=n_symbols).astype(np.uint8)

        config = GeneratorConfig(
            modulation="BPSK",
            num_symbols=n_symbols,
            sample_rate=10_000.0,
            symbol_rate=10_000.0,
            samples_per_symbol=1,
            random_seed=0,
        )

        # Generate noiseless symbols using the canonical ML modulator
        symbols = BPSKModulator().modulate(original_bits, config)

        # Demodulate with backend demodulator
        result = demod_bpsk(symbols)

        # Must recover every bit exactly
        np.testing.assert_array_equal(
            result.bits,
            original_bits,
            err_msg="BPSK round-trip failed: demodulated bits do not match original bits",
        )

    def test_qpsk_roundtrip_exact(self):
        """
        QPSK: bits → QPSKModulator symbols → demod_qpsk → same bits.
        Uses the ML QPSKModulator (canonical Gray mapping: b0b1 index=b0*2+b1).
        """
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

        from ml.generators.modulation.qpsk import QPSKModulator
        from ml.generators.config import GeneratorConfig

        rng = np.random.default_rng(1)
        n_symbols = 256
        # Must be even number of bits (2 per symbol)
        original_bits = rng.integers(0, 2, size=n_symbols * 2).astype(np.uint8)

        config = GeneratorConfig(
            modulation="QPSK",
            num_symbols=n_symbols,
            sample_rate=10_000.0,
            symbol_rate=10_000.0,
            samples_per_symbol=1,
            random_seed=1,
        )

        # Generate noiseless symbols using the canonical ML modulator
        symbols = QPSKModulator().modulate(original_bits, config)

        # Demodulate with backend demodulator
        result = demod_qpsk(symbols)

        # Must recover every bit exactly
        np.testing.assert_array_equal(
            result.bits,
            original_bits,
            err_msg="QPSK round-trip failed: demodulated bits do not match original bits",
        )

    def test_bpsk_roundtrip_all_bit_values(self):
        """All-zeros and all-ones bit sequences round-trip correctly."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

        from ml.generators.modulation.bpsk import BPSKModulator
        from ml.generators.config import GeneratorConfig

        config = GeneratorConfig(
            modulation="BPSK",
            num_symbols=8,
            sample_rate=1000.0,
            symbol_rate=1000.0,
            samples_per_symbol=1,
            random_seed=0,
        )
        mod = BPSKModulator()

        for constant_bit in (0, 1):
            bits = np.full(8, constant_bit, dtype=np.uint8)
            symbols = mod.modulate(bits, config)
            result = demod_bpsk(symbols)
            np.testing.assert_array_equal(result.bits, bits)

    def test_qpsk_roundtrip_all_dibits(self):
        """All four dibit patterns (00, 01, 10, 11) round-trip correctly."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

        from ml.generators.modulation.qpsk import QPSKModulator
        from ml.generators.config import GeneratorConfig

        config = GeneratorConfig(
            modulation="QPSK",
            num_symbols=4,
            sample_rate=1000.0,
            symbol_rate=1000.0,
            samples_per_symbol=1,
            random_seed=0,
        )
        mod = QPSKModulator()

        # One symbol per dibit: 00 01 10 11
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
        symbols = mod.modulate(bits, config)
        result = demod_qpsk(symbols)
        np.testing.assert_array_equal(result.bits, bits)
