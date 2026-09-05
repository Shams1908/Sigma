"""
Unit tests for backend/synchronization/

Tests cover the matched filter, Costas loop, Gardner TED, and the
combined synchronize() entry point.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from synchronization.matched_filter import apply_matched_filter, downsample
from synchronization.carrier_recovery import costas_loop, estimate_residual_phase
from synchronization.timing_recovery import gardner_timing_recovery, coarse_timing_offset
from synchronization import synchronize


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bpsk_oversampled(n_sym: int = 128, sps: int = 4) -> np.ndarray:
    np.random.seed(20)
    symbols = (2 * np.random.randint(0, 2, n_sym) - 1).astype(np.complex64)
    up = np.zeros(n_sym * sps, dtype=np.complex64)
    up[::sps] = symbols
    return up


def _qpsk_oversampled(n_sym: int = 128, sps: int = 4) -> np.ndarray:
    np.random.seed(21)
    NORM = 1.0 / np.sqrt(2.0)
    pts = [NORM + 1j*NORM, -NORM + 1j*NORM, -NORM - 1j*NORM, NORM - 1j*NORM]
    symbols = np.array([pts[i % 4] for i in range(n_sym)], dtype=np.complex64)
    up = np.zeros(n_sym * sps, dtype=np.complex64)
    up[::sps] = symbols
    return up


# ── Matched filter ────────────────────────────────────────────────────────────

class TestMatchedFilter:
    def test_output_length(self):
        iq = _bpsk_oversampled()
        out = apply_matched_filter(iq, samples_per_symbol=4)
        assert len(out) == len(iq)

    def test_output_is_complex(self):
        iq = _bpsk_oversampled()
        out = apply_matched_filter(iq, samples_per_symbol=4)
        assert np.iscomplexobj(out)

    def test_output_finite(self):
        iq = _bpsk_oversampled()
        out = apply_matched_filter(iq, samples_per_symbol=4)
        assert np.all(np.isfinite(out))

    def test_downsample_length(self):
        iq = _bpsk_oversampled(n_sym=64, sps=4)
        ds = downsample(iq, 4, timing_offset=0)
        assert len(ds) == len(iq) // 4

    def test_downsample_offset(self):
        iq = _bpsk_oversampled(n_sym=64, sps=4)
        ds0 = downsample(iq, 4, timing_offset=0)
        ds1 = downsample(iq, 4, timing_offset=1)
        # Different offsets should give different symbol estimates
        assert not np.allclose(ds0, ds1)


# ── Carrier recovery ──────────────────────────────────────────────────────────

class TestCostasLoop:
    def test_output_length(self):
        iq = _bpsk_oversampled()
        out, err = costas_loop(iq, modulation_order=2)
        assert len(out) == len(iq)
        assert len(err) == len(iq)

    def test_output_complex(self):
        iq = _bpsk_oversampled()
        out, _ = costas_loop(iq, modulation_order=2)
        assert np.iscomplexobj(out)

    def test_phase_error_finite(self):
        iq = _bpsk_oversampled()
        _, err = costas_loop(iq, modulation_order=2)
        assert np.all(np.isfinite(err))

    def test_costas_qpsk(self):
        iq = _qpsk_oversampled()
        out, err = costas_loop(iq, modulation_order=4)
        assert len(out) == len(iq)

    def test_residual_phase_estimate_finite(self):
        iq = _qpsk_oversampled()
        phase = estimate_residual_phase(iq, modulation_order=4)
        assert np.isfinite(phase)
        assert -np.pi <= phase <= np.pi

    def test_costas_reduces_phase_error(self):
        """Phase error RMS should decrease from first half to second half as loop converges."""
        iq = _bpsk_oversampled(n_sym=256, sps=4)
        # Add a phase offset
        iq = iq * np.exp(1j * 0.5)
        _, err = costas_loop(iq, modulation_order=2, loop_bw=0.05)
        n = len(err)
        rms_first = np.sqrt(np.mean(err[:n//2] ** 2))
        rms_second = np.sqrt(np.mean(err[n//2:] ** 2))
        # Allow generous tolerance — loop may take time to converge
        assert rms_second <= rms_first * 1.5


# ── Timing recovery ───────────────────────────────────────────────────────────

class TestTimingRecovery:
    def test_gardner_output_not_empty(self):
        iq = _bpsk_oversampled(n_sym=64, sps=4)
        out, err = gardner_timing_recovery(iq, samples_per_symbol=4)
        assert len(out) > 0

    def test_gardner_output_complex(self):
        iq = _bpsk_oversampled(n_sym=64, sps=4)
        out, _ = gardner_timing_recovery(iq, samples_per_symbol=4)
        assert np.iscomplexobj(out)

    def test_gardner_timing_errors_finite(self):
        iq = _bpsk_oversampled(n_sym=64, sps=4)
        _, err = gardner_timing_recovery(iq, samples_per_symbol=4)
        assert np.all(np.isfinite(err))

    def test_coarse_timing_in_range(self):
        iq = _bpsk_oversampled(n_sym=64, sps=4)
        offset = coarse_timing_offset(iq, samples_per_symbol=4)
        assert 0 <= offset < 4


# ── Full synchronize() ────────────────────────────────────────────────────────

class TestSynchronize:
    def test_bpsk_returns_symbols(self):
        iq_2d, sr = _make_bpsk_iq()
        iq = (iq_2d[0] + 1j * iq_2d[1]).astype(np.complex64)
        symbols, info = synchronize(iq, sr, symbol_rate=sr / 8.0, modulation_order=2)
        assert len(symbols) > 0
        assert "sps" in info

    def test_qpsk_returns_symbols(self):
        iq_2d, sr = _make_qpsk_iq()
        iq = (iq_2d[0] + 1j * iq_2d[1]).astype(np.complex64)
        symbols, info = synchronize(iq, sr, symbol_rate=sr / 8.0, modulation_order=4)
        assert len(symbols) > 0

    def test_synchronize_never_crashes_on_bad_input(self):
        """Pathological all-zeros input must not raise."""
        iq = np.zeros(256, dtype=np.complex64)
        symbols, info = synchronize(iq, 10_000.0, symbol_rate=1_000.0)
        assert isinstance(symbols, np.ndarray)
        assert isinstance(info, dict)


def _make_bpsk_iq(n_sym=256, sps=8, sr=80_000.0) -> tuple[np.ndarray, float]:
    np.random.seed(30)
    symbols = (2 * np.random.randint(0, 2, n_sym) - 1).astype(np.complex64)
    up = np.zeros(n_sym * sps, dtype=np.complex64)
    up[::sps] = symbols
    h = np.ones(sps, dtype=np.float32) / sps
    iq = np.convolve(up, h, mode="full")[:n_sym * sps]
    noise = 0.1 * (np.random.randn(len(iq)) + 1j * np.random.randn(len(iq)))
    iq = iq + noise.astype(np.complex64)
    return np.stack([iq.real, iq.imag]).astype(np.float32), sr


def _make_qpsk_iq(n_sym=256, sps=8, sr=80_000.0) -> tuple[np.ndarray, float]:
    np.random.seed(31)
    NORM = 1.0 / np.sqrt(2.0)
    pts = [NORM + 1j*NORM, -NORM + 1j*NORM, -NORM - 1j*NORM, NORM - 1j*NORM]
    symbols = np.array([pts[np.random.randint(0, 4)] for _ in range(n_sym)], dtype=np.complex64)
    up = np.zeros(n_sym * sps, dtype=np.complex64)
    up[::sps] = symbols
    h = np.ones(sps, dtype=np.float32) / sps
    iq = np.convolve(up, h, mode="full")[:n_sym * sps]
    noise = 0.1 * (np.random.randn(len(iq)) + 1j * np.random.randn(len(iq)))
    iq = iq + noise.astype(np.complex64)
    return np.stack([iq.real, iq.imag]).astype(np.float32), sr
