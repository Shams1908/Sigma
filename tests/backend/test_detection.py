"""
Unit tests for backend/dsp/detection.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure backend directory is in python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from dsp.detection import SignalRegion, detect_signal_regions


# Helper to generate synthetic signal
def create_synthetic_signal(
    sample_rate: float,
    num_samples: int,
    tone_freqs: list[float],
    snr_db: float = 20.0,
) -> np.ndarray:
    """
    Generate synthetic canonical IQ array [2, N] with tones at specific frequencies and AWGN.
    """
    np.random.seed(42)
    t = np.arange(num_samples) / sample_rate
    complex_sig = np.zeros(num_samples, dtype=np.complex64)

    for freq in tone_freqs:
        complex_sig += np.exp(1j * 2 * np.pi * freq * t).astype(np.complex64)

    if tone_freqs:
        # Scale tone power
        signal_pwr = 1.0
        noise_pwr = signal_pwr / (10.0 ** (snr_db / 10.0))
        noise_std = np.sqrt(noise_pwr / 2.0)
    else:
        noise_std = 0.1

    noise = (
        np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
    ) * noise_std

    total = complex_sig + noise
    iq_2d = np.stack([total.real, total.imag]).astype(np.float32)
    return iq_2d


# 1. Single strong tone in noise
def test_single_strong_tone():
    sr = 10_000.0
    tone_f = 2_000.0  # 2 kHz tone
    iq_2d = create_synthetic_signal(sr, 2048, [tone_f], snr_db=30.0)

    regions = detect_signal_regions(iq_2d, sr, threshold_db=10.0, min_bins=3)

    assert len(regions) >= 1
    # Find the region containing the tone frequency
    tone_region = None
    for r in regions:
        if r.start_frequency <= tone_f <= r.end_frequency:
            tone_region = r
            break

    assert tone_region is not None, f"Tone at {tone_freqs} not in any region: {regions}"
    assert np.isclose(tone_region.center_frequency, tone_f, atol=200.0)
    assert tone_region.snr_estimate_db > 10.0


# 2. Two separated signals
def test_two_separated_signals():
    sr = 20_000.0
    f1 = -4_000.0
    f2 = 3_000.0
    iq_2d = create_synthetic_signal(sr, 4096, [f1, f2], snr_db=25.0)

    regions = detect_signal_regions(iq_2d, sr, threshold_db=8.0, min_bins=3)

    assert len(regions) >= 2

    # Check that one region covers f1 and another covers f2
    r1_found = any(r.start_frequency <= f1 <= r.end_frequency for r in regions)
    r2_found = any(r.start_frequency <= f2 <= r.end_frequency for r in regions)

    assert r1_found, f"Frequency {f1} Hz not detected in regions: {regions}"
    assert r2_found, f"Frequency {f2} Hz not detected in regions: {regions}"


# 3. Noise-only input
def test_noise_only():
    sr = 10_000.0
    iq_2d = create_synthetic_signal(sr, 2048, tone_freqs=[], snr_db=0.0)

    # Pure noise should yield no regions above a high threshold (e.g. 15 dB)
    regions = detect_signal_regions(iq_2d, sr, threshold_db=15.0, min_bins=3)
    assert len(regions) == 0


# 4. Weak signal near threshold
def test_weak_signal_near_threshold():
    sr = 10_000.0
    tone_f = 1_500.0
    # Moderate SNR
    iq_2d = create_synthetic_signal(sr, 4096, [tone_f], snr_db=8.0)

    # Low threshold should detect it
    regions_low = detect_signal_regions(iq_2d, sr, threshold_db=4.0, min_bins=2)
    assert len(regions_low) >= 1

    # High threshold should miss it or return empty
    regions_high = detect_signal_regions(iq_2d, sr, threshold_db=25.0, min_bins=3)
    assert len(regions_high) == 0


# 5. Tiny isolated noise spike (filtered by min_bins)
def test_tiny_noise_spike_filtered():
    sr = 10_000.0
    iq_2d = create_synthetic_signal(sr, 2048, tone_freqs=[], snr_db=0.0)

    # Requiring min_bins=10 should filter small random fluctuations
    regions_strict = detect_signal_regions(iq_2d, sr, threshold_db=4.0, min_bins=10)
    regions_loose = detect_signal_regions(iq_2d, sr, threshold_db=4.0, min_bins=1)

    assert len(regions_strict) <= len(regions_loose)


# 6. Correct start/end frequency ordering
def test_frequency_ordering_and_positive_bandwidth():
    sr = 10_000.0
    tone_f = -2_000.0
    iq_2d = create_synthetic_signal(sr, 2048, [tone_f], snr_db=20.0)

    regions = detect_signal_regions(iq_2d, sr, threshold_db=6.0, min_bins=2)

    for r in regions:
        assert isinstance(r, SignalRegion)
        assert r.start_frequency <= r.end_frequency, (
            f"start_frequency ({r.start_frequency}) must be <= end_frequency ({r.end_frequency})"
        )
        assert r.bandwidth > 0, f"bandwidth ({r.bandwidth}) must be positive"


# 7. Correct approximate center frequency
def test_approximate_center_frequency():
    sr = 16_000.0
    target_f = 2_500.0
    iq_2d = create_synthetic_signal(sr, 4096, [target_f], snr_db=30.0)

    regions = detect_signal_regions(iq_2d, sr, threshold_db=10.0, min_bins=3)

    assert len(regions) >= 1
    best_region = min(regions, key=lambda r: abs(r.center_frequency - target_f))
    assert abs(best_region.center_frequency - target_f) < 250.0  # within 250 Hz


# 8. Invalid input handling
def test_invalid_inputs():
    sr = 10_000.0
    iq_valid = np.zeros((2, 100), dtype=np.float32)

    # Non-ndarray input
    with pytest.raises(TypeError):
        detect_signal_regions([[1, 2], [3, 4]], sr)  # type: ignore

    # Non-positive sample rate
    with pytest.raises(ValueError, match="Sample rate must be positive"):
        detect_signal_regions(iq_valid, 0.0)

    with pytest.raises(ValueError, match="Sample rate must be positive"):
        detect_signal_regions(iq_valid, -100.0)

    # Wrong shape (3D)
    with pytest.raises(ValueError):
        detect_signal_regions(np.zeros((2, 100, 2)), sr)

    # Non-finite values
    iq_nan = np.zeros((2, 100), dtype=np.float32)
    iq_nan[0, 5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        detect_signal_regions(iq_nan, sr)

    # Empty array
    with pytest.raises(ValueError):
        detect_signal_regions(np.zeros((2, 0)), sr)


# 9. Direct gap merging unit test
def test_merge_spectral_gaps_direct():
    from dsp.detection import merge_spectral_gaps

    # Mask: 1 1 1 0 0 1 1 1 0 0 0 0 1 1
    # Gaps: 2-bin gap (indices 3..4), 4-bin gap (indices 8..11)
    mask = np.array([True, True, True, False, False, True, True, True, False, False, False, False, True, True])
    merged = merge_spectral_gaps(mask, merge_gap_bins=2)

    expected = np.array([True, True, True, True, True, True, True, True, False, False, False, False, True, True])
    assert np.array_equal(merged, expected)


# 10. Direct contiguous regions unit test
def test_find_contiguous_regions_direct():
    from dsp.detection import find_contiguous_regions

    mask = np.array([False, True, True, True, False, False, True, True])
    regions = find_contiguous_regions(mask)

    assert regions == [(1, 3), (6, 7)]


# 11. Preprocessing segmentation package import check
def test_preprocessing_segmentation_imports():
    from preprocessing.segmentation import (
        SignalRegion as PrepSignalRegion,
        detect_signal_regions as prep_detect,
        find_contiguous_regions as prep_find,
        merge_spectral_gaps as prep_merge,
    )

    assert PrepSignalRegion is SignalRegion
    assert prep_detect is detect_signal_regions

