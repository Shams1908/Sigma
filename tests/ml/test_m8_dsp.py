"""
Unit tests for M8 Deterministic DSP Parameter Estimators (ml/parameters/dsp_estimators.py).
"""
import pytest
import numpy as np

from ml.parameters.dsp_estimators import (
    estimate_symbol_rate_dsp,
    estimate_snr_dsp,
    DSPSymbolRateResult,
    DSPSNRResult,
    _ensure_complex_1d,
)
from ml.parameters.dataset import synthesize_parameter_sample


def test_ensure_complex_1d():
    """Verifies complex input conversion and shape validation."""
    # 2D [2, N] float
    iq_2d = np.ones((2, 100), dtype=np.float32)
    c = _ensure_complex_1d(iq_2d)
    assert c.ndim == 1
    assert len(c) == 100
    assert np.iscomplexobj(c)

    # 1D complex
    c_in = np.ones(50, dtype=np.complex64)
    c_out = _ensure_complex_1d(c_in)
    assert len(c_out) == 50

    # Invalid shape
    with pytest.raises(ValueError):
        _ensure_complex_1d(np.ones((3, 50)))


def test_dsp_symbol_rate_clean_bpsk():
    """Verifies DSP symbol rate estimation on clean BPSK with known Baud rate."""
    sample_rate = 800000.0
    true_baud = 50000.0  # 50 kBaud
    sps = sample_rate / true_baud  # 16 sps

    # Generate 4096 samples (~256 symbols) for clean spectral line
    sample = synthesize_parameter_sample(
        modulation="BPSK",
        sps=sps,
        snr_db=30.0,
        sample_rate=sample_rate,
        length=4096,
        apply_multipath=False,
    )

    res = estimate_symbol_rate_dsp(sample, sample_rate=sample_rate)
    assert isinstance(res, DSPSymbolRateResult)
    assert res.confidence > 0.60
    assert res.uncertainty > 0.0
    # Tolerance within ±5%
    rel_err = abs(res.estimate - true_baud) / true_baud
    assert rel_err < 0.05, f"Expected ~{true_baud}, got {res.estimate} (rel_err: {rel_err:.3f})"


def test_dsp_snr_estimation():
    """Verifies M2M4 moment estimator tracks SNR trends."""
    sample_rate = 800000.0
    sps = 8.0

    # High SNR sample (25 dB)
    sample_high = synthesize_parameter_sample(
        modulation="QPSK", sps=sps, snr_db=25.0, sample_rate=sample_rate, length=2048
    )
    res_high = estimate_snr_dsp(sample_high)

    # Low SNR sample (5 dB)
    sample_low = synthesize_parameter_sample(
        modulation="QPSK", sps=sps, snr_db=5.0, sample_rate=sample_rate, length=2048
    )
    res_low = estimate_snr_dsp(sample_low)

    assert isinstance(res_high, DSPSNRResult)
    assert isinstance(res_low, DSPSNRResult)
    # High SNR estimate must exceed Low SNR estimate
    assert res_high.estimate > res_low.estimate
    assert res_high.confidence > 0.40
