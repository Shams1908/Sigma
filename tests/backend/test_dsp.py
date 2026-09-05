"""
Unit tests for backend/dsp/*.

Tests validate that each estimator returns plausible numeric results
for synthetic BPSK and QPSK signals — not that they hit exact values,
but that they are within physically reasonable bounds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from dsp.fft import canonical_to_complex, compute_fft, peak_frequency
from dsp.psd import estimate_psd, noise_floor_db, signal_power_db
from dsp.snr import estimate_snr, _m2m4_snr
from dsp.bandwidth import estimate_bandwidth
from dsp.carrier import estimate_carrier_offset, remove_carrier_offset
from dsp.symbol_rate import estimate_symbol_rate
from dsp import extract_parameters


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def bpsk_signal():
    """Return (iq_2d [2,N], sample_rate) for a clean BPSK signal."""
    np.random.seed(0)
    sps = 8
    n_sym = 512
    sr = 10_000.0
    symbols = (2 * np.random.randint(0, 2, n_sym) - 1).astype(np.complex64)
    upsampled = np.zeros(n_sym * sps, dtype=np.complex64)
    upsampled[::sps] = symbols
    h = np.ones(sps, dtype=np.float32) / sps
    iq = np.convolve(upsampled, h, mode="full")[:n_sym * sps]
    # Add small noise so PSD estimators have something to work with
    noise = 0.05 * (np.random.randn(len(iq)) + 1j * np.random.randn(len(iq)))
    iq = iq + noise.astype(np.complex64)
    iq_2d = np.stack([iq.real, iq.imag]).astype(np.float32)
    return iq_2d, sr


@pytest.fixture(scope="module")
def complex_iq(bpsk_signal):
    iq_2d, sr = bpsk_signal
    return canonical_to_complex(iq_2d), sr


# ── canonical_to_complex ──────────────────────────────────────────────────────

def test_canonical_to_complex_shape(bpsk_signal):
    iq_2d, _ = bpsk_signal
    iq = canonical_to_complex(iq_2d)
    assert iq.ndim == 1
    assert len(iq) == iq_2d.shape[1]
    assert np.iscomplexobj(iq)


def test_canonical_to_complex_nrow_layout():
    """Also accept [N, 2] layout."""
    arr = np.random.randn(100, 2).astype(np.float32)
    iq = canonical_to_complex(arr)
    assert len(iq) == 100


def test_canonical_to_complex_bad_shape():
    with pytest.raises(ValueError):
        canonical_to_complex(np.zeros((3, 100)))


# ── FFT ───────────────────────────────────────────────────────────────────────

def test_compute_fft_length(complex_iq):
    iq, sr = complex_iq
    freqs, spec = compute_fft(iq, sr)
    assert len(freqs) == len(spec)
    assert len(freqs) > 0


def test_compute_fft_empty():
    with pytest.raises(ValueError):
        compute_fft(np.array([], dtype=np.complex64), 10_000.0)


def test_peak_frequency_dc():
    """Pure DC signal → peak at 0 Hz."""
    n = 1024
    iq = np.ones(n, dtype=np.complex64)
    sr = 10_000.0
    freqs, spec = compute_fft(iq, sr)
    peak = peak_frequency(freqs, spec)
    assert abs(peak) < sr / n * 2, f"DC peak should be near 0 Hz, got {peak}"


# ── PSD ───────────────────────────────────────────────────────────────────────

def test_estimate_psd_returns_finite(complex_iq):
    iq, sr = complex_iq
    freqs, psd = estimate_psd(iq, sr)
    assert np.all(np.isfinite(psd))
    assert len(freqs) == len(psd)


def test_noise_floor_lower_than_signal(complex_iq):
    iq, sr = complex_iq
    _, psd = estimate_psd(iq, sr)
    floor = noise_floor_db(psd)
    sig = signal_power_db(psd, floor)
    assert sig >= floor


def test_psd_short_signal():
    """Short signal should not crash — falls back to smaller nperseg."""
    iq = (np.random.randn(30) + 1j * np.random.randn(30)).astype(np.complex64)
    freqs, psd = estimate_psd(iq, 10_000.0)
    assert len(psd) > 0


# ── SNR ───────────────────────────────────────────────────────────────────────

def test_snr_reasonable_range(complex_iq):
    iq, sr = complex_iq
    snr = estimate_snr(iq, sr)
    assert -10 <= snr <= 60, f"SNR out of range: {snr}"


def test_snr_higher_for_cleaner_signal():
    """Higher-SNR signal should produce a larger (or equal) SNR estimate."""
    np.random.seed(1)
    sr = 10_000.0
    clean = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64)
    noisy = clean + 2.0 * (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64)
    snr_clean = estimate_snr(clean, sr)
    snr_noisy = estimate_snr(noisy, sr)
    # Clean should have higher or equal SNR estimate
    assert snr_clean >= snr_noisy - 3.0  # 3 dB tolerance for estimation error


def test_m2m4_fallback():
    """M2M4 estimator should return a finite value."""
    np.random.seed(2)
    iq = (np.random.randn(64) + 1j * np.random.randn(64)).astype(np.complex64)
    snr = _m2m4_snr(iq)
    assert np.isfinite(snr)
    assert -10 <= snr <= 60


# ── Bandwidth ─────────────────────────────────────────────────────────────────

def test_bandwidth_positive(complex_iq):
    iq, sr = complex_iq
    bw = estimate_bandwidth(iq, sr)
    assert bw > 0, "Bandwidth must be positive"
    assert bw <= sr, "Bandwidth cannot exceed sample rate"


def test_bandwidth_wideband_vs_narrowband():
    """
    Wider signal should have a larger bandwidth estimate than a narrowband signal.
    Both signals are generated as filtered noise to ensure the PSD estimator has
    power distributed across different frequency spans.
    """
    np.random.seed(3)
    sr = 40_000.0
    n = 4096
    from scipy.signal import firwin, lfilter

    noise = np.random.randn(n) + 1j * np.random.randn(n)
    noise = noise.astype(np.complex64)

    # Wide: pass everything below 15 kHz  (nyquist = 20 kHz)
    cutoff_wide = 15_000.0 / (sr / 2.0)
    h_wide = firwin(31, cutoff_wide).astype(np.float32)
    wide = lfilter(h_wide, 1.0, noise).astype(np.complex64)

    # Narrow: pass only up to 2 kHz
    cutoff_narrow = 2_000.0 / (sr / 2.0)
    h_narrow = firwin(31, cutoff_narrow).astype(np.float32)
    narrow = lfilter(h_narrow, 1.0, noise).astype(np.complex64)

    bw_wide = estimate_bandwidth(wide, sr)
    bw_narrow = estimate_bandwidth(narrow, sr)
    assert bw_wide >= bw_narrow, (
        f"Wide BW {bw_wide:.0f} Hz should be >= narrow BW {bw_narrow:.0f} Hz"
    )


# ── Carrier offset ────────────────────────────────────────────────────────────

def test_carrier_offset_returns_finite(complex_iq):
    iq, sr = complex_iq
    cfo = estimate_carrier_offset(iq, sr)
    assert np.isfinite(cfo)


def test_remove_carrier_offset_reduces_residual():
    """After correction, the CFO of the corrected signal should be smaller."""
    np.random.seed(4)
    sr = 10_000.0
    true_cfo = 1_000.0  # 1 kHz
    n = 2048
    baseband = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)
    t = np.arange(n) / sr
    with_cfo = baseband * np.exp(1j * 2 * np.pi * true_cfo * t).astype(np.complex64)

    estimated_cfo = estimate_carrier_offset(with_cfo, sr)
    corrected = remove_carrier_offset(with_cfo, sr, estimated_cfo)
    residual_cfo = abs(estimate_carrier_offset(corrected, sr))
    original_cfo_magnitude = abs(estimated_cfo)

    # Correction should reduce the offset (within estimation noise)
    assert residual_cfo <= original_cfo_magnitude + 200.0, (
        f"Residual CFO {residual_cfo:.1f} Hz is not reduced from {original_cfo_magnitude:.1f} Hz"
    )


# ── Symbol rate ───────────────────────────────────────────────────────────────

def test_symbol_rate_returns_positive(complex_iq):
    iq, sr = complex_iq
    rs = estimate_symbol_rate(iq, sr)
    assert rs > 0


def test_symbol_rate_in_plausible_range(bpsk_signal):
    """Symbol rate estimate should be within 3× of the true rate."""
    iq_2d, sr = bpsk_signal
    true_sr = sr / 8.0  # sps=8
    iq = canonical_to_complex(iq_2d)
    estimated = estimate_symbol_rate(iq, sr, min_rate=100.0)
    ratio = max(estimated, true_sr) / max(min(estimated, true_sr), 1.0)
    assert ratio <= 3.0, (
        f"Symbol rate estimate {estimated:.0f} too far from true {true_sr:.0f}"
    )


# ── extract_parameters (integration) ─────────────────────────────────────────

def test_extract_parameters_keys(bpsk_signal):
    iq_2d, sr = bpsk_signal
    params = extract_parameters(iq_2d, sr)
    for key in ("snr", "carrier_offset", "bandwidth", "symbol_rate_estimate"):
        assert key in params, f"Missing key: {key}"
        assert np.isfinite(params[key]), f"Non-finite value for {key}: {params[key]}"


def test_extract_parameters_never_crashes():
    """extract_parameters must not raise even for pathological input."""
    # All-zeros signal
    iq_2d = np.zeros((2, 128), dtype=np.float32)
    params = extract_parameters(iq_2d, 10_000.0)
    assert isinstance(params, dict)
