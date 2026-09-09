"""
Tests for newly implemented demodulators:
  8PSK  (backend/demodulation/psk8_demod.py)
  16-QAM / 64-QAM  (backend/demodulation/qam_demod.py)
  2-FSK / 4-FSK    (backend/demodulation/fsk_demod.py)

Coverage:
  - Generator round-trip (exact bit recovery on clean symbols) where an ML
    generator exists (8PSK, 16QAM, 64QAM).
  - All-symbols round-trip: every constellation point exactly decoded.
  - Noisy signal: EVM / detection still passes at SNR ≥ 10 dB.
  - Edge cases: empty input, single symbol.
  - Dispatcher routing via demodulate().
  - Existing BPSK/QPSK dispatcher routes unchanged (regression).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from demodulation.psk8_demod import demod_8psk, _8PSK_CONST, _8PSK_BITS_TABLE
from demodulation.qam_demod import (
    demod_qam16, demod_qam64,
    _QAM16_CONST, _QAM16_ALL_BITS,
    _QAM64_CONST, _QAM64_ALL_BITS,
    QAMDemodNotImplemented,
)
from demodulation.fsk_demod import demod_2fsk, demod_4fsk
from demodulation import demodulate, DemodResult, constellation_score


# ── Helpers ───────────────────────────────────────────────────────────────────

def _add_awgn(symbols: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """Add complex AWGN to unit-power symbols."""
    rng = np.random.default_rng(seed)
    p_signal = float(np.mean(np.abs(symbols) ** 2))
    p_noise = p_signal / (10.0 ** (snr_db / 10.0))
    std = np.sqrt(p_noise / 2.0)
    noise = (rng.normal(0, std, len(symbols))
             + 1j * rng.normal(0, std, len(symbols))).astype(np.complex64)
    return symbols + noise


def _fsk_tone_signal(
    bits: np.ndarray,
    tone_freqs_rad: list[float],
    sps: int = 8,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic FSK signal: each bit/dibit selects a tone from
    tone_freqs_rad (radians/sample), repeated for sps samples.
    Returns (complex_iq, tone_freq_array) where the second is one entry
    per sample (for reference).
    """
    m = len(tone_freqs_rad)  # 2 or 4
    bits_per_sym = 1 if m == 2 else 2
    n_sym = len(bits) // bits_per_sym

    # tone index per symbol
    if m == 2:
        tone_idx = bits[:n_sym]
    else:  # 4FSK: Gray reverse — convert bits to tone_idx
        _GRAY_INV = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}
        pairs = bits[:n_sym * 2].reshape(-1, 2)
        tone_idx = np.array([_GRAY_INV[tuple(p)] for p in pairs.tolist()])

    samples = []
    phase = 0.0
    for ti in tone_idx:
        f = tone_freqs_rad[int(ti)]
        for _ in range(sps):
            samples.append(np.exp(1j * phase).astype(np.complex64))
            phase += f
    return np.array(samples, dtype=np.complex64), tone_idx


# ── 8PSK ──────────────────────────────────────────────────────────────────────

class TestDemod8PSK:
    def test_result_type(self):
        r = demod_8psk(_8PSK_CONST[:4])
        assert isinstance(r, DemodResult)
        assert r.modulation == "8PSK"

    def test_bits_per_symbol(self):
        n = 32
        syms = _8PSK_CONST[np.arange(n) % 8]
        r = demod_8psk(syms)
        assert len(r.bits) == 3 * n

    def test_bits_are_binary(self):
        r = demod_8psk(_8PSK_CONST)
        assert set(r.bits.tolist()).issubset({0, 1})

    def test_all_constellation_points_decoded_exactly(self):
        """Each of the 8 constellation points must decode to its exact 3-bit word."""
        for idx in range(8):
            sym = np.array([_8PSK_CONST[idx]], dtype=np.complex64)
            r = demod_8psk(sym)
            expected = _8PSK_BITS_TABLE[idx].tolist()
            assert r.bits.tolist() == expected, (
                f"idx={idx}: got {r.bits.tolist()}, expected {expected}"
            )

    def test_clean_evm_near_zero(self):
        syms = _8PSK_CONST  # exact constellation points
        r = demod_8psk(syms)
        assert r.evm_rms < 1e-5

    def test_noisy_evm_below_threshold(self):
        syms = _8PSK_CONST[np.arange(64) % 8]
        noisy = _add_awgn(syms, snr_db=15.0)
        r = demod_8psk(noisy)
        assert r.evm_rms < 0.5

    def test_empty_input(self):
        r = demod_8psk(np.array([], dtype=np.complex64))
        assert len(r.bits) == 0
        assert r.evm_rms == 0.0

    def test_single_symbol(self):
        # index 0 (000): angle=0 → (1, 0)
        r = demod_8psk(np.array([1.0 + 0.0j], dtype=np.complex64))
        assert list(r.bits) == [0, 0, 0]

    def test_generator_roundtrip(self):
        """Symbols from ML EightPSKModulator decode back to the original bits."""
        from ml.generators.modulation.psk8 import EightPSKModulator
        from ml.generators.config import GeneratorConfig

        rng = np.random.default_rng(42)
        n_sym = 128
        original_bits = rng.integers(0, 2, n_sym * 3).astype(np.uint8)

        config = GeneratorConfig(
            modulation="8PSK",
            num_symbols=n_sym,
            sample_rate=10_000.0,
            symbol_rate=10_000.0,
            samples_per_symbol=1,
            random_seed=42,
        )
        symbols = EightPSKModulator().modulate(original_bits, config)
        r = demod_8psk(symbols)
        np.testing.assert_array_equal(r.bits, original_bits)

    def test_generator_roundtrip_all_dibits(self):
        """All 8 possible 3-bit patterns round-trip through ML generator exactly."""
        from ml.generators.modulation.psk8 import EightPSKModulator
        from ml.generators.config import GeneratorConfig

        config = GeneratorConfig(
            modulation="8PSK",
            num_symbols=8,
            sample_rate=1000.0,
            symbol_rate=1000.0,
            samples_per_symbol=1,
            random_seed=0,
        )
        mod = EightPSKModulator()
        # One symbol per 3-bit pattern 000..111
        bits = np.array([b for i in range(8) for b in [int(c) for c in f"{i:03b}"]],
                        dtype=np.uint8)
        symbols = mod.modulate(bits, config)
        r = demod_8psk(symbols)
        np.testing.assert_array_equal(r.bits, bits)


# ── 16-QAM ────────────────────────────────────────────────────────────────────

class TestDemod16QAM:
    def test_result_type(self):
        r = demod_qam16(_QAM16_CONST[:4])
        assert isinstance(r, DemodResult)
        assert r.modulation == "QAM16"

    def test_bits_per_symbol(self):
        r = demod_qam16(_QAM16_CONST)   # 16 symbols
        assert len(r.bits) == 4 * 16

    def test_bits_are_binary(self):
        r = demod_qam16(_QAM16_CONST)
        assert set(r.bits.tolist()).issubset({0, 1})

    def test_all_16_points_decoded_exactly(self):
        """All 16 constellation points decode to their exact 4-bit words."""
        for k in range(16):
            sym = np.array([_QAM16_CONST[k]], dtype=np.complex64)
            r = demod_qam16(sym)
            expected = _QAM16_ALL_BITS[k].tolist()
            assert r.bits.tolist() == expected, (
                f"k={k}: got {r.bits.tolist()}, expected {expected}"
            )

    def test_clean_evm_near_zero(self):
        r = demod_qam16(_QAM16_CONST)
        assert r.evm_rms < 1e-5

    def test_noisy_evm_below_threshold(self):
        syms = _QAM16_CONST[np.arange(64) % 16]
        noisy = _add_awgn(syms, snr_db=20.0)
        r = demod_qam16(noisy)
        assert r.evm_rms < 0.5

    def test_empty_input(self):
        r = demod_qam16(np.array([], dtype=np.complex64))
        assert len(r.bits) == 0

    def test_generator_roundtrip(self):
        """Symbols from ML QAM16Modulator decode back to original bits."""
        from ml.generators.modulation.qam16 import QAM16Modulator
        from ml.generators.config import GeneratorConfig

        rng = np.random.default_rng(7)
        n_sym = 64
        original_bits = rng.integers(0, 2, n_sym * 4).astype(np.uint8)

        config = GeneratorConfig(
            modulation="QAM16",
            num_symbols=n_sym,
            sample_rate=10_000.0,
            symbol_rate=10_000.0,
            samples_per_symbol=1,
            random_seed=7,
        )
        symbols = QAM16Modulator().modulate(original_bits, config)
        r = demod_qam16(symbols)
        np.testing.assert_array_equal(r.bits, original_bits)

    def test_generator_roundtrip_all_16_patterns(self):
        """All 16 4-bit patterns round-trip through ML generator."""
        from ml.generators.modulation.qam16 import QAM16Modulator
        from ml.generators.config import GeneratorConfig

        config = GeneratorConfig(
            modulation="QAM16",
            num_symbols=16,
            sample_rate=1000.0,
            symbol_rate=1000.0,
            samples_per_symbol=1,
            random_seed=0,
        )
        bits = np.array(
            [b for i in range(16) for b in [int(c) for c in f"{i:04b}"]],
            dtype=np.uint8,
        )
        symbols = QAM16Modulator().modulate(bits, config)
        r = demod_qam16(symbols)
        np.testing.assert_array_equal(r.bits, bits)


# ── 64-QAM ────────────────────────────────────────────────────────────────────

class TestDemod64QAM:
    def test_result_type(self):
        r = demod_qam64(_QAM64_CONST[:4])
        assert isinstance(r, DemodResult)
        assert r.modulation == "QAM64"

    def test_bits_per_symbol(self):
        r = demod_qam64(_QAM64_CONST)   # 64 symbols
        assert len(r.bits) == 6 * 64

    def test_bits_are_binary(self):
        r = demod_qam64(_QAM64_CONST)
        assert set(r.bits.tolist()).issubset({0, 1})

    def test_all_64_points_decoded_exactly(self):
        """All 64 constellation points decode to their exact 6-bit words."""
        for k in range(64):
            sym = np.array([_QAM64_CONST[k]], dtype=np.complex64)
            r = demod_qam64(sym)
            expected = _QAM64_ALL_BITS[k].tolist()
            assert r.bits.tolist() == expected, (
                f"k={k}: got {r.bits.tolist()}, expected {expected}"
            )

    def test_clean_evm_near_zero(self):
        r = demod_qam64(_QAM64_CONST)
        assert r.evm_rms < 1e-5

    def test_noisy_low_ber_high_snr(self):
        """At SNR=25 dB, 64-QAM should have negligible bit errors."""
        rng = np.random.default_rng(99)
        n = 256
        symbols = _QAM64_CONST[rng.integers(0, 64, n)]
        noisy = _add_awgn(symbols, snr_db=25.0, seed=99)
        r_clean = demod_qam64(symbols)
        r_noisy = demod_qam64(noisy)
        total = len(r_clean.bits)
        errors = int(np.sum(r_clean.bits != r_noisy.bits))
        ber = errors / total
        assert ber < 0.05, f"BER={ber:.4f} too high at SNR=25 dB"

    def test_empty_input(self):
        r = demod_qam64(np.array([], dtype=np.complex64))
        assert len(r.bits) == 0

    def test_generator_roundtrip(self):
        """Symbols from ML QAM64Modulator decode back to original bits."""
        from ml.generators.modulation.qam64 import QAM64Modulator
        from ml.generators.config import GeneratorConfig

        rng = np.random.default_rng(11)
        n_sym = 64
        original_bits = rng.integers(0, 2, n_sym * 6).astype(np.uint8)

        config = GeneratorConfig(
            modulation="QAM64",
            num_symbols=n_sym,
            sample_rate=10_000.0,
            symbol_rate=10_000.0,
            samples_per_symbol=1,
            random_seed=11,
        )
        symbols = QAM64Modulator().modulate(original_bits, config)
        r = demod_qam64(symbols)
        np.testing.assert_array_equal(r.bits, original_bits)


# ── 2-FSK ─────────────────────────────────────────────────────────────────────

class TestDemod2FSK:
    _TONES = [-0.1, 0.1]  # radians/sample: low=0, high=1
    _SPS = 8

    def _make_signal(self, bits: np.ndarray, snr_db: float | None = None) -> np.ndarray:
        sig, _ = _fsk_tone_signal(bits, self._TONES, sps=self._SPS)
        if snr_db is not None:
            sig = _add_awgn(sig, snr_db)
        return sig

    def test_result_type(self):
        sig = self._make_signal(np.array([0, 1, 0, 1], dtype=np.uint8))
        r = demod_2fsk(sig, sps=self._SPS, tone_freqs=tuple(self._TONES))
        assert isinstance(r, DemodResult)
        assert r.modulation == "2FSK"

    def test_evm_is_zero(self):
        """FSK EVM is always 0.0 (not applicable)."""
        sig = self._make_signal(np.array([0, 1, 0, 1], dtype=np.uint8))
        r = demod_2fsk(sig, sps=self._SPS, tone_freqs=tuple(self._TONES))
        assert r.evm_rms == 0.0

    def test_bits_per_symbol(self):
        bits = np.array([0, 1, 0, 0, 1, 1, 0, 1], dtype=np.uint8)
        sig = self._make_signal(bits)
        r = demod_2fsk(sig, sps=self._SPS, tone_freqs=tuple(self._TONES))
        assert len(r.bits) == len(bits)

    def test_exact_recovery_noiseless(self):
        """Known bit sequence must round-trip exactly (noiseless)."""
        rng = np.random.default_rng(0)
        bits = rng.integers(0, 2, 32).astype(np.uint8)
        sig = self._make_signal(bits)
        r = demod_2fsk(sig, sps=self._SPS, tone_freqs=tuple(self._TONES))
        np.testing.assert_array_equal(r.bits, bits)

    def test_all_zeros(self):
        sig = self._make_signal(np.zeros(8, dtype=np.uint8))
        r = demod_2fsk(sig, sps=self._SPS, tone_freqs=tuple(self._TONES))
        assert np.all(r.bits == 0)

    def test_all_ones(self):
        sig = self._make_signal(np.ones(8, dtype=np.uint8))
        r = demod_2fsk(sig, sps=self._SPS, tone_freqs=tuple(self._TONES))
        assert np.all(r.bits == 1)

    def test_noisy_low_ber(self):
        """At SNR=15 dB, 2-FSK BER should be very low."""
        rng = np.random.default_rng(5)
        bits = rng.integers(0, 2, 128).astype(np.uint8)
        sig = self._make_signal(bits, snr_db=15.0)
        r = demod_2fsk(sig, sps=self._SPS, tone_freqs=tuple(self._TONES))
        ber = float(np.mean(r.bits != bits))
        assert ber < 0.05, f"2FSK BER={ber:.4f} at SNR=15 dB"

    def test_empty_input(self):
        r = demod_2fsk(np.array([], dtype=np.complex64), sps=1)
        assert len(r.bits) == 0

    def test_no_tone_freqs_auto_threshold(self):
        """Without explicit tone_freqs, auto-threshold should still work."""
        rng = np.random.default_rng(3)
        bits = rng.integers(0, 2, 64).astype(np.uint8)
        sig = self._make_signal(bits)
        r = demod_2fsk(sig, sps=self._SPS)   # no tone_freqs
        # With balanced bits, auto-threshold should give low BER
        ber = float(np.mean(r.bits != bits))
        assert ber < 0.15  # generous — auto-threshold is approximate

    def test_dispatcher_routing(self):
        sig = self._make_signal(np.array([0, 1, 0, 1], dtype=np.uint8))
        r = demodulate(sig, "2FSK", sps=self._SPS, tone_freqs=tuple(self._TONES))
        assert r.modulation == "2FSK"


# ── 4-FSK ─────────────────────────────────────────────────────────────────────

class TestDemod4FSK:
    _TONES = [-0.15, -0.05, 0.05, 0.15]   # 4 equally spaced tones rad/sample
    _SPS = 8

    def _make_signal(
        self,
        bits: np.ndarray,
        snr_db: float | None = None,
        seed: int = 42,
    ) -> np.ndarray:
        sig, _ = _fsk_tone_signal(bits, self._TONES, sps=self._SPS, seed=seed)
        if snr_db is not None:
            sig = _add_awgn(sig, snr_db, seed=seed)
        return sig

    def test_result_type(self):
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
        sig = self._make_signal(bits)
        r = demod_4fsk(sig, sps=self._SPS, tone_freqs=tuple(self._TONES))
        assert isinstance(r, DemodResult)
        assert r.modulation == "4FSK"

    def test_bits_per_symbol(self):
        bits = np.array([0, 0, 0, 1, 1, 1, 1, 0] * 4, dtype=np.uint8)
        sig = self._make_signal(bits)
        r = demod_4fsk(sig, sps=self._SPS, tone_freqs=tuple(self._TONES))
        assert len(r.bits) == len(bits)

    def test_bits_are_binary(self):
        bits = np.array([0, 0, 0, 1, 1, 1, 1, 0] * 4, dtype=np.uint8)
        sig = self._make_signal(bits)
        r = demod_4fsk(sig, sps=self._SPS, tone_freqs=tuple(self._TONES))
        assert set(r.bits.tolist()).issubset({0, 1})

    def test_exact_recovery_noiseless(self):
        """Known dibit sequence must round-trip exactly."""
        rng = np.random.default_rng(1)
        bits = rng.integers(0, 2, 64).astype(np.uint8)
        sig = self._make_signal(bits)
        r = demod_4fsk(sig, sps=self._SPS, tone_freqs=tuple(self._TONES))
        np.testing.assert_array_equal(r.bits, bits)

    def test_all_dibits(self):
        """All four Gray-coded dibit patterns decode correctly."""
        # dibits: 00 01 10 11 — each is one symbol with tone 0,1,3,2 respectively
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
        sig = self._make_signal(bits)
        r = demod_4fsk(sig, sps=self._SPS, tone_freqs=tuple(self._TONES))
        np.testing.assert_array_equal(r.bits, bits)

    def test_noisy_low_ber(self):
        rng = np.random.default_rng(9)
        bits = rng.integers(0, 2, 128).astype(np.uint8)
        sig = self._make_signal(bits, snr_db=20.0, seed=9)
        r = demod_4fsk(sig, sps=self._SPS, tone_freqs=tuple(self._TONES))
        ber = float(np.mean(r.bits != bits))
        assert ber < 0.05, f"4FSK BER={ber:.4f} at SNR=20 dB"

    def test_empty_input(self):
        r = demod_4fsk(np.array([], dtype=np.complex64), sps=1)
        assert len(r.bits) == 0

    def test_invalid_tone_freqs_length(self):
        with pytest.raises(ValueError, match="4 elements"):
            demod_4fsk(np.ones(16, dtype=np.complex64), sps=1,
                       tone_freqs=(0.1, 0.2, 0.3))  # only 3

    def test_dispatcher_routing(self):
        bits = np.array([0, 0, 1, 1] * 8, dtype=np.uint8)
        sig = self._make_signal(bits)
        r = demodulate(sig, "4FSK", sps=self._SPS, tone_freqs=tuple(self._TONES))
        assert r.modulation == "4FSK"


# ── Dispatcher tests ──────────────────────────────────────────────────────────

class TestDispatcher:
    def test_8psk_routing(self):
        r = demodulate(_8PSK_CONST[:8], "8PSK")
        assert r.modulation == "8PSK"

    def test_16qam_routing(self):
        r = demodulate(_QAM16_CONST[:4], "16QAM")
        assert r.modulation == "QAM16"

    def test_16qam_alias_qam16(self):
        r = demodulate(_QAM16_CONST[:4], "QAM16")
        assert r.modulation == "QAM16"

    def test_64qam_routing(self):
        r = demodulate(_QAM64_CONST[:4], "64QAM")
        assert r.modulation == "QAM64"

    def test_64qam_alias_qam64(self):
        r = demodulate(_QAM64_CONST[:4], "QAM64")
        assert r.modulation == "QAM64"

    def test_unsupported_raises(self):
        with pytest.raises(ValueError):
            demodulate(np.zeros(4, dtype=np.complex64), "AM-DSB")

    # Regression: existing BPSK/QPSK routes still work
    def test_bpsk_still_routes(self):
        syms = np.array([1.0 + 0j, -1.0 + 0j], dtype=np.complex64)
        r = demodulate(syms, "BPSK")
        assert r.modulation == "BPSK"

    def test_qpsk_still_routes(self):
        from demodulation.psk_demod import _QPSK_NORM
        sym = np.array([_QPSK_NORM + 1j * _QPSK_NORM], dtype=np.complex64)
        r = demodulate(sym, "QPSK")
        assert r.modulation == "QPSK"


# ── constellation_score compatibility ─────────────────────────────────────────

class TestConstellationScore:
    def test_8psk_score_range(self):
        syms = _8PSK_CONST[np.arange(16) % 8]
        noisy = _add_awgn(syms, snr_db=15.0)
        r = demod_8psk(noisy)
        score = constellation_score(r)
        assert 0.0 <= score <= 1.0

    def test_16qam_score_clean(self):
        r = demod_qam16(_QAM16_CONST)
        assert constellation_score(r) > 0.9

    def test_fsk_score_is_zero_since_evm_zero(self):
        """FSK EVM=0 → score=1.0 (technically maximal, though not meaningful)."""
        bits = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
        sig, _ = _fsk_tone_signal(bits, [-0.1, 0.1], sps=4)
        r = demod_2fsk(sig, sps=4, tone_freqs=(-0.1, 0.1))
        assert constellation_score(r) == 1.0
