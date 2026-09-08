"""
Unit tests for signal extraction and filtering (backend/preprocessing/filtering.py & backend/dsp/extraction.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure backend directory is in python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from dsp.detection import SignalRegion
from dsp.fft import compute_fft, peak_frequency, canonical_to_complex
from preprocessing.filtering import ExtractedSignal, extract_signal, lowpass_filter


# Helper to generate synthetic tone
def generate_synthetic_tones(
    sample_rate: float,
    num_samples: int,
    tone_freqs: list[float],
    snr_db: float = 25.0,
) -> np.ndarray:
    np.random.seed(42)
    t = np.arange(num_samples) / sample_rate
    sig = np.zeros(num_samples, dtype=np.complex64)
    for freq in tone_freqs:
        sig += np.exp(1j * 2 * np.pi * freq * t).astype(np.complex64)

    noise_std = 0.05
    noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * noise_std
    total = sig + noise
    return np.stack([total.real, total.imag]).astype(np.float32)


# 1. Single signal extraction & sign convention check
def test_extract_single_tone_baseband_shift():
    sr = 100_000.0
    n = 4096
    tone_f = 10_000.0  # +10 kHz
    iq_2d = generate_synthetic_tones(sr, n, [tone_f])

    region = SignalRegion(
        start_frequency=9_000.0,
        end_frequency=11_000.0,
        center_frequency=tone_f,
        bandwidth=2_000.0,
        peak_power_db=0.0,
        average_power_db=-3.0,
        snr_estimate_db=20.0,
    )

    extracted = extract_signal(iq_2d, sr, region)
    assert isinstance(extracted, ExtractedSignal)

    complex_out = canonical_to_complex(extracted.iq)
    freqs, spec = compute_fft(complex_out, sr)
    dominant_f = peak_frequency(freqs, spec)

    # Dominant frequency should be shifted from +10 kHz down to ~0 Hz
    assert abs(dominant_f) < 200.0, f"Expected center near 0 Hz, got {dominant_f:.1f} Hz"


# 2. Two signals: target signal centered at 0 Hz, interferer attenuated
def test_extract_two_signals_interferer_attenuation():
    sr = 100_000.0
    n = 8192
    tone_a = -15_000.0  # Interferer
    tone_b = 20_000.0   # Target signal
    iq_2d = generate_synthetic_tones(sr, n, [tone_a, tone_b])

    region_b = SignalRegion(
        start_frequency=18_000.0,
        end_frequency=22_000.0,
        center_frequency=tone_b,
        bandwidth=4_000.0,
        peak_power_db=0.0,
        average_power_db=-3.0,
        snr_estimate_db=20.0,
    )

    extracted = extract_signal(iq_2d, sr, region_b)
    complex_out = canonical_to_complex(extracted.iq)

    freqs, spec = compute_fft(complex_out, sr)

    # Target Tone B should be at ~0 Hz
    dominant_f = peak_frequency(freqs, spec)
    assert abs(dominant_f) < 250.0, f"Target tone B should be near 0 Hz, got {dominant_f:.1f} Hz"

    # Interferer Tone A (originally -15 kHz, shifted by -20 kHz to -35 kHz)
    # Power at 0 Hz should be significantly higher than power at -35 kHz after LPF
    idx_dc = np.argmin(np.abs(freqs - 0.0))
    idx_interferer = np.argmin(np.abs(freqs - (-35_000.0)))
    power_dc = spec[idx_dc]
    power_interferer = spec[idx_interferer]

    assert power_dc > power_interferer + 15.0, (
        f"Interferer power ({power_interferer:.1f} dB) should be >15 dB below target ({power_dc:.1f} dB)"
    )


# 3. Output format compliance
def test_extracted_signal_output_format():
    sr = 50_000.0
    n = 1024
    iq_2d = generate_synthetic_tones(sr, n, [5_000.0])

    region = SignalRegion(
        start_frequency=4_000.0,
        end_frequency=6_000.0,
        center_frequency=5_000.0,
        bandwidth=2_000.0,
        peak_power_db=0.0,
        average_power_db=-3.0,
        snr_estimate_db=15.0,
    )

    extracted = extract_signal(iq_2d, sr, region)

    assert isinstance(extracted.iq, np.ndarray)
    assert extracted.iq.ndim == 2
    assert extracted.iq.shape == (2, n)
    assert extracted.iq.dtype == np.float32
    assert np.all(np.isfinite(extracted.iq))
    assert extracted.sample_rate == sr  # Original sample rate preserved (no resampling)
    assert extracted.original_region == region


# 4. Invalid input handling
def test_extract_signal_invalid_inputs():
    sr = 10_000.0
    region_valid = SignalRegion(
        start_frequency=1_000.0,
        end_frequency=3_000.0,
        center_frequency=2_000.0,
        bandwidth=2_000.0,
        peak_power_db=0.0,
        average_power_db=0.0,
        snr_estimate_db=10.0,
    )

    # Invalid non-SignalRegion
    with pytest.raises(TypeError):
        extract_signal(np.zeros((2, 100)), sr, "not_a_region")  # type: ignore

    # Invalid negative bandwidth region
    bad_region = SignalRegion(1000, 3000, 2000, -500.0, 0, 0, 0)
    with pytest.raises(ValueError, match="bandwidth must be positive"):
        extract_signal(np.zeros((2, 100)), sr, bad_region)

    # Invalid 3D IQ input
    with pytest.raises(ValueError):
        extract_signal(np.zeros((2, 100, 2)), sr, region_valid)

    # Non-positive sample rate
    with pytest.raises(ValueError, match="Sample rate must be positive"):
        lowpass_filter(np.zeros((2, 100)), 0.0, 1000.0)


# 5. Edge frequencies near Nyquist
def test_edge_frequencies_near_nyquist():
    sr = 100_000.0
    n = 2048
    tone_f = 42_000.0  # Near Nyquist (50 kHz)
    iq_2d = generate_synthetic_tones(sr, n, [tone_f])

    region = SignalRegion(
        start_frequency=40_000.0,
        end_frequency=44_000.0,
        center_frequency=tone_f,
        bandwidth=4_000.0,
        peak_power_db=0.0,
        average_power_db=-3.0,
        snr_estimate_db=15.0,
    )

    extracted = extract_signal(iq_2d, sr, region)
    assert extracted.iq.shape == (2, n)
    assert np.all(np.isfinite(extracted.iq))


# 6. Zero signal handling
def test_zero_signal_extraction():
    sr = 10_000.0
    iq_zero = np.zeros((2, 512), dtype=np.float32)

    region = SignalRegion(
        start_frequency=1_000.0,
        end_frequency=3_000.0,
        center_frequency=2_000.0,
        bandwidth=2_000.0,
        peak_power_db=-300.0,
        average_power_db=-300.0,
        snr_estimate_db=0.0,
    )

    extracted = extract_signal(iq_zero, sr, region)
    assert np.all(extracted.iq == 0.0)
    assert np.all(np.isfinite(extracted.iq))
    assert extracted.iq.shape == (2, 512)


# 7. Explicit sign convention validation
def test_sign_convention_verification():
    """
    Explicit test verifying that multiplying by exp(-j * 2pi * f_center * t)
    shifts a positive tone at +f_center down to 0 Hz.
    """
    sr = 20_000.0
    n = 2048
    f_pos = 5_000.0
    iq_2d = generate_synthetic_tones(sr, n, [f_pos])

    region = SignalRegion(
        start_frequency=4_000.0,
        end_frequency=6_000.0,
        center_frequency=f_pos,
        bandwidth=2_000.0,
        peak_power_db=0.0,
        average_power_db=0.0,
        snr_estimate_db=20.0,
    )

    extracted = extract_signal(iq_2d, sr, region)
    complex_sig = canonical_to_complex(extracted.iq)
    freqs, spec = compute_fft(complex_sig, sr)

    # Shifted peak should be at 0 Hz
    peak_f = peak_frequency(freqs, spec)
    assert abs(peak_f) < 50.0, f"Expected 0 Hz after shift, got {peak_f} Hz"
