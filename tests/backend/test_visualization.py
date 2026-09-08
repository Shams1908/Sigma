"""
Unit tests for DSP visualization data generation (backend/dsp/visualization.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure backend directory is in python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from dsp.detection import SignalRegion
from dsp.visualization import (
    generate_waveform_data,
    generate_fft_data,
    generate_psd_data,
    generate_spectrogram_data,
    generate_detection_data,
    generate_parameter_data,
    generate_visualization_data,
    to_json_serializable,
    downsample_for_visualization,
)


def generate_synthetic_iq(sample_rate: float, num_samples: int, tone_freq: float = 10_000.0) -> np.ndarray:
    np.random.seed(42)
    t = np.arange(num_samples) / sample_rate
    sig = np.exp(1j * 2 * np.pi * tone_freq * t).astype(np.complex64)
    noise = 0.05 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)).astype(np.complex64)
    total = sig + noise
    return np.stack([total.real, total.imag]).astype(np.float32)


# 1. Waveform data generation
def test_generate_waveform_data():
    sr = 50_000.0
    n = 5000
    iq_2d = generate_synthetic_iq(sr, n)

    wave = generate_waveform_data(iq_2d, sr, max_points=1000)

    assert "time" in wave and "i" in wave and "q" in wave
    assert len(wave["time"]) == len(wave["i"]) == len(wave["q"])
    assert len(wave["time"]) <= 1000
    assert wave["decimation"] >= 5

    # Verify JSON serializability
    serialized = json.dumps(wave)
    assert isinstance(serialized, str)


# 2. FFT data generation & peak frequency matching
def test_generate_fft_data():
    sr = 100_000.0
    n = 2048
    tone_f = 15_000.0
    iq_2d = generate_synthetic_iq(sr, n, tone_freq=tone_f)

    fft_data = generate_fft_data(iq_2d, sr, max_points=1000)

    assert "frequency" in fft_data and "magnitude_db" in fft_data
    assert len(fft_data["frequency"]) == len(fft_data["magnitude_db"])
    assert len(fft_data["frequency"]) <= 1000

    # Dominant peak frequency check
    freqs = np.array(fft_data["frequency"])
    mags = np.array(fft_data["magnitude_db"])
    peak_f = freqs[np.argmax(mags)]

    assert abs(peak_f - tone_f) < 500.0, f"Expected peak near {tone_f} Hz, got {peak_f} Hz"

    # Verify JSON serializability
    json.dumps(fft_data)


# 3. PSD data generation
def test_generate_psd_data():
    sr = 40_000.0
    n = 2048
    iq_2d = generate_synthetic_iq(sr, n)

    psd_data = generate_psd_data(iq_2d, sr, max_points=500)

    assert "frequency" in psd_data and "power_db" in psd_data
    assert len(psd_data["frequency"]) == len(psd_data["power_db"])
    assert len(psd_data["frequency"]) <= 500
    assert np.all(np.isfinite(psd_data["power_db"]))

    json.dumps(psd_data)


# 4. STFT spectrogram data generation
def test_generate_spectrogram_data():
    sr = 20_000.0
    n = 4096
    iq_2d = generate_synthetic_iq(sr, n)

    stft_data = generate_spectrogram_data(
        iq_2d,
        sr,
        max_time_points=100,
        max_freq_points=100,
    )

    assert "frequency" in stft_data and "time" in stft_data and "spectrogram_db" in stft_data
    n_freqs = len(stft_data["frequency"])
    n_times = len(stft_data["time"])
    matrix = stft_data["spectrogram_db"]

    assert len(matrix) == n_freqs
    assert len(matrix[0]) == n_times

    json.dumps(stft_data)


# 5. Detected signal region visualization
def test_generate_detection_data():
    regions = [
        SignalRegion(
            start_frequency=1000.0,
            end_frequency=5000.0,
            center_frequency=3000.0,
            bandwidth=4000.0,
            peak_power_db=5.0,
            average_power_db=2.0,
            snr_estimate_db=18.5,
        )
    ]

    det_data = generate_detection_data(regions)

    assert isinstance(det_data, list)
    assert len(det_data) == 1
    item = det_data[0]
    assert item["start_frequency"] == 1000.0
    assert item["end_frequency"] == 5000.0
    assert item["center_frequency"] == 3000.0
    assert item["bandwidth"] == 4000.0
    assert item["peak_power_db"] == 5.0

    json.dumps(det_data)


# 6. Parameter estimation visualization & None preservation
def test_generate_parameter_data():
    # Test with numeric SNR and None symbol rate
    params = {
        "snr": 12.5,
        "bandwidth": 3500.0,
        "carrier_offset": 120.0,
        "symbol_rate_estimate": None,
    }

    param_data = generate_parameter_data(params)

    assert param_data["snr"] == 12.5
    assert param_data["bandwidth"] == 3500.0
    assert param_data["carrier_offset"] == 120.0
    assert param_data["symbol_rate"] is None  # None preserved honestly, no fake 0

    json.dumps(param_data)


# 7. Large signal downsampling
def test_large_signal_downsampling():
    sr = 1_000_000.0
    n_large = 200_000  # 200k samples
    iq_large = generate_synthetic_iq(sr, n_large)

    wave = generate_waveform_data(iq_large, sr, max_points=1000)

    assert len(wave["time"]) <= 1000
    assert len(wave["i"]) <= 1000

    # Test standalone helper
    arr = np.linspace(0, 100, 5000)
    downsampled = downsample_for_visualization(arr, max_points=500)
    assert len(downsampled) <= 500
    assert downsampled[0] == arr[0]


# 8. Composite JSON serialization validation
def test_composite_visualization_json_serialization():
    sr = 50_000.0
    iq_2d = generate_synthetic_iq(sr, 4096)

    regions = [
        SignalRegion(1000.0, 3000.0, 2000.0, 2000.0, 0.0, -3.0, 15.0)
    ]
    params = {
        "snr": 15.0,
        "bandwidth": 2000.0,
        "carrier_offset": 50.0,
        "symbol_rate_estimate": None,
    }

    composite = generate_visualization_data(
        iq_2d,
        sr,
        include_waveform=True,
        include_fft=True,
        include_psd=True,
        include_spectrogram=True,
        regions=regions,
        params=params,
        max_points=500,
    )

    assert "waveform" in composite
    assert "fft" in composite
    assert "psd" in composite
    assert "spectrogram" in composite
    assert "detected_regions" in composite
    assert "parameters" in composite

    # Must be completely JSON serializable without throwing TypeError
    raw_json = json.dumps(composite)
    assert isinstance(raw_json, str)
    parsed = json.loads(raw_json)
    assert parsed["parameters"]["symbol_rate"] is None
