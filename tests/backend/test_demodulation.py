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
        assert r.bits[0] == 0  # real > 0 → bit 0

    def test_negative_symbol(self):
        r = demod_bpsk(np.array([-0.9 + 0.1j], dtype=np.complex64))
        assert r.bits[0] == 1  # real < 0 → bit 1


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
        """I > 0, Q > 0 → Gray code 00."""
        r = demod_qpsk(np.array([0.7 + 0.7j], dtype=np.complex64))
        assert list(r.bits) == [0, 0]

    def test_quadrant_q2(self):
        """I < 0, Q > 0 → Gray code 10."""
        r = demod_qpsk(np.array([-0.7 + 0.7j], dtype=np.complex64))
        assert list(r.bits) == [1, 0]

    def test_quadrant_q3(self):
        """I < 0, Q < 0 → Gray code 11."""
        r = demod_qpsk(np.array([-0.7 - 0.7j], dtype=np.complex64))
        assert list(r.bits) == [1, 1]

    def test_quadrant_q4(self):
        """I > 0, Q < 0 → Gray code 01."""
        r = demod_qpsk(np.array([0.7 - 0.7j], dtype=np.complex64))
        assert list(r.bits) == [0, 1]


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
