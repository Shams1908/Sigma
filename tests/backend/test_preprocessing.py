"""
Unit tests for backend/preprocessing/normalize.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure backend directory is in python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from preprocessing.normalize import (
    validate_canonical_iq,
    remove_dc,
    normalize_power,
    normalize_peak,
    normalize_signal,
)


# 1. Valid canonical IQ input
def test_valid_canonical_iq():
    iq = np.random.randn(2, 100).astype(np.float32)
    validated = validate_canonical_iq(iq)
    assert np.array_equal(validated, iq)


# 2. Wrong input type
def test_wrong_input_type():
    with pytest.raises(TypeError, match="Input must be a NumPy ndarray"):
        validate_canonical_iq([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(TypeError, match="Input must be a NumPy ndarray"):
        validate_canonical_iq("not_an_array")


# 3. Wrong dimensions
def test_wrong_dimensions():
    # 1D array
    with pytest.raises(ValueError, match="Input must have exactly 2 dimensions"):
        validate_canonical_iq(np.array([1.0, 2.0, 3.0]))

    # 3D array
    with pytest.raises(ValueError, match="Input must have exactly 2 dimensions"):
        validate_canonical_iq(np.zeros((2, 100, 2)))


# 4. Wrong shape where first dimension is not 2
def test_wrong_shape_first_dim():
    # [3, 100]
    with pytest.raises(ValueError, match="First dimension of canonical IQ array must be 2"):
        validate_canonical_iq(np.zeros((3, 100)))

    # [100, 2] (transposed / non-canonical)
    with pytest.raises(ValueError, match="First dimension of canonical IQ array must be 2"):
        validate_canonical_iq(np.zeros((100, 2)))


# 5. Empty signal
def test_empty_signal():
    with pytest.raises(ValueError, match="must contain at least one sample"):
        validate_canonical_iq(np.zeros((2, 0)))


# 6. NaN values
def test_nan_values():
    iq = np.zeros((2, 100), dtype=np.float32)
    iq[0, 10] = np.nan
    with pytest.raises(ValueError, match="contains non-finite values"):
        validate_canonical_iq(iq)


# 7. Inf values
def test_inf_values():
    iq = np.zeros((2, 100), dtype=np.float32)
    iq[1, 50] = np.inf
    with pytest.raises(ValueError, match="contains non-finite values"):
        validate_canonical_iq(iq)

    iq[1, 50] = -np.inf
    with pytest.raises(ValueError, match="contains non-finite values"):
        validate_canonical_iq(iq)


# 8. DC offset removal
def test_dc_offset_removal():
    np.random.seed(42)
    # Signal with known DC offset (I + 5.0, Q - 3.0)
    i = np.random.randn(500) + 5.0
    q = np.random.randn(500) - 3.0
    iq = np.stack([i, q]).astype(np.float32)

    iq_no_dc = remove_dc(iq)

    assert iq_no_dc.shape == (2, 500)
    assert iq_no_dc.dtype == np.float32
    assert np.isclose(np.mean(iq_no_dc[0]), 0.0, atol=1e-5)
    assert np.isclose(np.mean(iq_no_dc[1]), 0.0, atol=1e-5)


# 9. Unit-power normalization
def test_unit_power_normalization():
    np.random.seed(42)
    # Create arbitrary non-zero signal
    iq = (np.random.randn(2, 1000) * 10.0).astype(np.float32)

    iq_norm = normalize_power(iq)

    assert iq_norm.shape == (2, 1000)
    assert iq_norm.dtype == np.float32
    mean_power = np.mean(iq_norm[0] ** 2 + iq_norm[1] ** 2)
    assert np.isclose(mean_power, 1.0, atol=1e-4)


# 10. Zero signal normalization
def test_zero_signal_normalization():
    iq_zero = np.zeros((2, 100), dtype=np.float32)

    # remove_dc
    res_dc = remove_dc(iq_zero)
    assert np.all(res_dc == 0.0)
    assert res_dc.dtype == np.float32
    assert res_dc.shape == (2, 100)

    # normalize_power
    res_pow = normalize_power(iq_zero)
    assert np.all(res_pow == 0.0)
    assert res_pow.dtype == np.float32
    assert res_pow.shape == (2, 100)
    assert np.all(np.isfinite(res_pow))

    # normalize_peak
    res_peak = normalize_peak(iq_zero)
    assert np.all(res_peak == 0.0)
    assert res_peak.dtype == np.float32
    assert res_peak.shape == (2, 100)
    assert np.all(np.isfinite(res_peak))

    # normalize_signal
    res_sig = normalize_signal(iq_zero)
    assert np.all(res_sig == 0.0)
    assert res_sig.dtype == np.float32
    assert res_sig.shape == (2, 100)
    assert np.all(np.isfinite(res_sig))


# 11. Peak normalization
def test_peak_normalization():
    np.random.seed(42)
    iq = (np.random.randn(2, 500) * 5.0).astype(np.float32)

    iq_peak = normalize_peak(iq)

    assert iq_peak.shape == (2, 500)
    assert iq_peak.dtype == np.float32
    max_mag = np.max(np.abs(iq_peak))
    assert np.isclose(max_mag, 1.0, atol=1e-5)


# 12. Original input is not modified
def test_original_input_not_modified():
    iq_orig = np.array([[10.0, 20.0, 30.0], [5.0, 15.0, 25.0]], dtype=np.float64)
    iq_copy = iq_orig.copy()

    _ = remove_dc(iq_orig)
    assert np.array_equal(iq_orig, iq_copy)

    _ = normalize_power(iq_orig)
    assert np.array_equal(iq_orig, iq_copy)

    _ = normalize_peak(iq_orig)
    assert np.array_equal(iq_orig, iq_copy)

    _ = normalize_signal(iq_orig)
    assert np.array_equal(iq_orig, iq_copy)


# 13. Output dtype is float32
def test_output_dtype_float32():
    iq_float64 = np.random.randn(2, 200).astype(np.float64)

    assert remove_dc(iq_float64).dtype == np.float32
    assert normalize_power(iq_float64).dtype == np.float32
    assert normalize_peak(iq_float64).dtype == np.float32
    assert normalize_signal(iq_float64).dtype == np.float32


# 14. Output shape remains [2, N]
def test_output_shape_remains_2_n():
    n_samples = 345
    iq = np.random.randn(2, n_samples).astype(np.float32)

    assert remove_dc(iq).shape == (2, n_samples)
    assert normalize_power(iq).shape == (2, n_samples)
    assert normalize_peak(iq).shape == (2, n_samples)
    assert normalize_signal(iq).shape == (2, n_samples)
