"""
Unit and Integration tests for the DSP Pipeline Orchestrator.
"""

from __future__ import annotations

import json
import pytest
import numpy as np

from dsp.pipeline import (
    DSPPipelineConfig,
    RegionResult,
    DSPPipelineResult,
    run_dsp_pipeline,
)
from dsp.detection import SignalRegion


def make_synthetic_tone(
    freq_hz: float,
    sample_rate: float = 100000.0,
    duration_sec: float = 0.05,
    noise_std: float = 0.05,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate canonical IQ array [2, N] for a complex exponential tone with noise."""
    n_samples = int(sample_rate * duration_sec)
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    tone = amplitude * np.exp(1j * 2.0 * np.pi * freq_hz * t)
    
    np.random.seed(42)
    noise_i = np.random.normal(0, noise_std, n_samples)
    noise_q = np.random.normal(0, noise_std, n_samples)
    
    i_samples = tone.real + noise_i
    q_samples = tone.imag + noise_q
    
    return np.array([i_samples, q_samples], dtype=np.float32)


def test_single_signal_end_to_end():
    """TEST 1 — Single signal end-to-end processing."""
    sample_rate = 100000.0
    tone_freq = 10000.0
    iq = make_synthetic_tone(freq_hz=tone_freq, sample_rate=sample_rate, noise_std=0.05)
    
    config = DSPPipelineConfig(
        enable_normalization=True,
        enable_fft=True,
        enable_psd=True,
        generate_visualization=True,
        extract_detected_signals=True,
    )
    
    result = run_dsp_pipeline(iq, sample_rate, config=config)
    
    # 1. Normalization completes
    assert result.normalized_signal.shape == (2, iq.shape[1])
    assert np.all(np.isfinite(result.normalized_signal))
    
    # 2. PSD & FFT generated
    assert result.psd_data is not None
    freqs_p, power_db = result.psd_data
    assert len(freqs_p) == len(power_db)
    
    assert result.fft_data is not None
    freqs_f, mag_db = result.fft_data
    assert len(freqs_f) == len(mag_db)
    
    # 3. Detection near 10000 Hz
    assert len(result.detected_regions) >= 1
    detected_center = result.detected_regions[0].center_frequency
    assert abs(detected_center - tone_freq) < 1500.0  # within tolerance
    
    # 4. Parameter estimation produced
    assert len(result.region_results) == len(result.detected_regions)
    rr0 = result.region_results[0]
    assert rr0.parameters["center_frequency"] == pytest.approx(detected_center, abs=10.0)
    assert rr0.parameters["snr"] > 0.0
    
    # 5. Extraction produced
    assert rr0.extracted_signal is not None
    assert rr0.extracted_signal.iq.ndim == 2
    assert rr0.extracted_signal.iq.shape[0] == 2
    assert rr0.extracted_signal.iq.shape[1] > 0
    
    # 6. Visualization data valid
    assert result.visualization_data is not None
    assert "waveform" in result.visualization_data
    assert "psd" in result.visualization_data
    assert "detected_regions" in result.visualization_data


def test_two_signals_end_to_end():
    """TEST 2 — Two signals end-to-end processing."""
    sample_rate = 100000.0
    sig1 = make_synthetic_tone(freq_hz=-15000.0, sample_rate=sample_rate, noise_std=0.02)
    sig2 = make_synthetic_tone(freq_hz=20000.0, sample_rate=sample_rate, noise_std=0.02)
    
    iq_two = sig1 + sig2
    
    config = DSPPipelineConfig(
        threshold_db=5.0,
        extract_detected_signals=True,
    )
    
    result = run_dsp_pipeline(iq_two, sample_rate, config=config)
    
    # Multiple regions detected
    assert len(result.detected_regions) >= 2
    assert len(result.region_results) == len(result.detected_regions)
    
    centers = [r.center_frequency for r in result.detected_regions]
    has_neg15k = any(abs(c - (-15000.0)) < 2500.0 for c in centers)
    has_pos20k = any(abs(c - 20000.0) < 2500.0 for c in centers)
    
    assert has_neg15k, f"Expected region near -15000 Hz, got centers: {centers}"
    assert has_pos20k, f"Expected region near +20000 Hz, got centers: {centers}"
    
    for rr in result.region_results:
        assert rr.parameters is not None
        assert rr.extracted_signal is not None


def test_noise_only():
    """TEST 3 — Noise-only input handling."""
    sample_rate = 100000.0
    np.random.seed(123)
    noise_i = np.random.normal(0, 0.01, 2000).astype(np.float32)
    noise_q = np.random.normal(0, 0.01, 2000).astype(np.float32)
    iq_noise = np.array([noise_i, noise_q], dtype=np.float32)
    
    config = DSPPipelineConfig(threshold_db=15.0)  # high threshold to guarantee no detection
    result = run_dsp_pipeline(iq_noise, sample_rate, config=config)
    
    assert result.detected_regions == []
    assert result.region_results == []
    assert result.visualization_data is not None


def test_zero_signal():
    """TEST 4 — Zero-signal input handling."""
    sample_rate = 100000.0
    iq_zero = np.zeros((2, 2000), dtype=np.float32)
    
    result = run_dsp_pipeline(iq_zero, sample_rate)
    
    assert result.detected_regions == []
    assert result.region_results == []
    assert np.all(np.isfinite(result.normalized_signal))


def test_json_serialization():
    """TEST 5 & 8 — Output consistency & JSON serialization."""
    sample_rate = 100000.0
    iq = make_synthetic_tone(freq_hz=5000.0, sample_rate=sample_rate)
    
    config = DSPPipelineConfig(enable_stft=True, generate_visualization=True)
    result = run_dsp_pipeline(iq, sample_rate, config=config)
    
    result_dict = result.to_dict(include_visualization=True)
    
    # Verify json.dumps succeeds without error
    json_str = json.dumps(result_dict)
    assert isinstance(json_str, str)
    assert len(json_str) > 0


def test_stft_configuration():
    """Verify STFT spectrogram is generated when requested, omitted by default."""
    sample_rate = 100000.0
    iq = make_synthetic_tone(freq_hz=5000.0, sample_rate=sample_rate)
    
    # Default (stft=False)
    r_default = run_dsp_pipeline(iq, sample_rate)
    assert r_default.stft_data is None
    
    # Enabled (stft=True)
    config_stft = DSPPipelineConfig(enable_stft=True)
    r_stft = run_dsp_pipeline(iq, sample_rate, config=config_stft)
    assert r_stft.stft_data is not None
    times, freqs, stft_db = r_stft.stft_data
    assert stft_db.shape == (len(freqs), len(times))


def test_invalid_input_raises():
    """Verify pipeline raises appropriate exceptions on invalid input."""
    # Negative or zero sample rate
    iq = np.zeros((2, 100), dtype=np.float32)
    with pytest.raises(ValueError, match="Sample rate must be positive"):
        run_dsp_pipeline(iq, 0.0)

    # Empty array
    iq_empty = np.array([], dtype=np.complex64)
    with pytest.raises(ValueError, match="at least one sample"):
        run_dsp_pipeline(iq_empty, 10000.0)
