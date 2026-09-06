import os
import tempfile
import pickle
import pytest
import numpy as np

from ml.features.schema import FEATURE_NAMES, NUM_FEATURES
from ml.features.amplitude import extract_amplitude_features
from ml.features.phase import extract_phase_features
from ml.features.frequency import extract_frequency_features
from ml.features.cumulants import extract_cumulant_features
from ml.features.correlation import extract_autocorrelation_features
from ml.features.extractor import (
    extract_features,
    extract_batch_features,
    extract_feature_matrix,
    get_feature_names,
)
from ml.dataset.loader import RadioMLDataset, DatasetCache
from ml.dataset.labels import MODULATION_CLASSES

# 1. Test Amplitude Feature Family
def test_amplitude_features():
    # Test case 1: Constant amplitude signal x[n] = 2.0 * exp(1j * theta)
    # Amplitude a[n] = 2.0
    theta = np.linspace(0, 2*np.pi, 128)
    x = 2.0 * np.exp(1j * theta)
    
    feats = extract_amplitude_features(x)
    assert np.allclose(feats["amplitude_mean"], 2.0, atol=1e-5)
    assert np.allclose(feats["amplitude_variance"], 0.0, atol=1e-5)
    assert np.allclose(feats["amplitude_kurtosis"], 0.0, atol=1e-5) # variance is 0 -> fallback to 0
    assert np.allclose(feats["amplitude_peak_to_average_ratio"], 1.0, atol=1e-5) # peak=2, mean_power=4 -> 4/4 = 1.0
    
    # Test case 2: Fluctuating amplitude
    # x has amplitudes [1.0, 3.0, 1.0, 3.0, ...]
    a_pattern = np.array([1.0, 3.0] * 64, dtype=np.float32)
    x_fluc = a_pattern * np.exp(1j * theta)
    feats_fluc = extract_amplitude_features(x_fluc)
    
    # mean = 2.0, var = 1.0
    assert np.allclose(feats_fluc["amplitude_mean"], 2.0, atol=1e-5)
    assert np.allclose(feats_fluc["amplitude_variance"], 1.0, atol=1e-5)
    
    # Kurtosis of [1, 3] pattern: mean=2, diff=[-1, 1], m4=1, var=1 -> 1/1 - 3 = -2.0
    assert np.allclose(feats_fluc["amplitude_kurtosis"], -2.0, atol=1e-5)
    
    # PAR: max=3, mean(a^2) = (1 + 9)/2 = 5.0 -> PAR = 9 / 5 = 1.8
    assert np.allclose(feats_fluc["amplitude_peak_to_average_ratio"], 1.8, atol=1e-5)

    # Test case 3: Zero-power signal handling
    x_zero = np.zeros(128, dtype=np.complex64)
    feats_zero = extract_amplitude_features(x_zero)
    assert feats_zero["amplitude_mean"] == 0.0
    assert feats_zero["amplitude_variance"] == 0.0
    assert feats_zero["amplitude_kurtosis"] == 0.0
    assert feats_zero["amplitude_peak_to_average_ratio"] == 0.0
    assert not np.isnan(list(feats_zero.values())).any()
    assert np.isfinite(list(feats_zero.values())).all()

# 2. Test Phase Feature Family
def test_phase_features():
    # Test case 1: Constant phase signal x[n] = exp(1j * pi / 4)
    # phase variance = 0
    theta = np.ones(128) * (np.pi / 4.0)
    x = np.exp(1j * theta)
    
    feats = extract_phase_features(x)
    assert np.allclose(feats["phase_variance"], 0.0, atol=1e-5)
    assert np.allclose(feats["phase_difference_mean"], 0.0, atol=1e-5)
    assert np.allclose(feats["phase_difference_variance"], 0.0, atol=1e-5)
    assert np.allclose(feats["phase_difference_kurtosis"], 0.0, atol=1e-5)
    
    # Histogram sum check
    hist_sum = sum(feats[f"phase_hist_bin_{k}"] for k in range(8))
    assert np.allclose(hist_sum, 1.0, atol=1e-5)
    
    # Check that phase pi/4 falls into bin 4 ([0, pi/4)) or bin 5 ([pi/4, pi/2))
    # pi/4 normalized value: (pi/4 + pi)/(2*pi) * 8 = 1.25/2 * 8 = 5.0 -> bin 5
    assert np.allclose(feats["phase_hist_bin_5"], 1.0, atol=1e-5)
    assert np.allclose(feats["phase_hist_bin_0"], 0.0, atol=1e-5)

    # Test case 2: Zero signal phase safety
    x_zero = np.zeros(128, dtype=np.complex64)
    feats_zero = extract_phase_features(x_zero)
    assert feats_zero["phase_variance"] == 0.0
    assert feats_zero["phase_difference_mean"] == 0.0
    assert feats_zero["phase_difference_variance"] == 0.0
    assert feats_zero["phase_difference_kurtosis"] == 0.0
    # histogram should sum to 1 (all phases forced to 0.0 -> bin 4)
    hist_sum_zero = sum(feats_zero[f"phase_hist_bin_{k}"] for k in range(8))
    assert np.allclose(hist_sum_zero, 1.0, atol=1e-5)
    assert feats_zero["phase_hist_bin_4"] == 1.0
    assert not np.isnan(list(feats_zero.values())).any()
    assert np.isfinite(list(feats_zero.values())).all()

# 3. Test Frequency Feature Family
def test_frequency_features():
    # Test case 1: Complex exponential x[n] = exp(1j * omega * n)
    # Phase difference (frequency) dphi = omega
    omega = 0.5
    n = np.arange(128)
    x = np.exp(1j * omega * n)
    
    feats = extract_frequency_features(x)
    assert np.allclose(feats["instantaneous_frequency_mean"], omega, atol=1e-5)
    assert np.allclose(feats["instantaneous_frequency_variance"], 0.0, atol=1e-5)
    
    # Test case 2: Constant phase signal
    x_const = np.ones(128, dtype=np.complex64)
    feats_const = extract_frequency_features(x_const)
    assert feats_const["instantaneous_frequency_mean"] == 0.0
    assert feats_const["instantaneous_frequency_variance"] == 0.0

# 4. Test Cumulant Feature Family
def test_cumulant_features():
    # Test case 1: Constant signal x[n] = 1.0
    # E[x^4] = 1, E[x^2] = 1, E[|x|^2] = 1
    # C40_raw = 1 - 3*(1)^2 = -2.0
    # Normalized C40 = -2.0
    x = np.ones(128, dtype=np.complex64)
    feats = extract_cumulant_features(x)
    assert np.allclose(feats["c40_real"], -2.0, atol=1e-5)
    assert np.allclose(feats["c40_imag"], 0.0, atol=1e-5)
    assert np.allclose(feats["c40_magnitude"], 2.0, atol=1e-5)
    
    # Test case 2: Zero power signal handling
    x_zero = np.zeros(128, dtype=np.complex64)
    feats_zero = extract_cumulant_features(x_zero)
    assert feats_zero["c40_real"] == 0.0
    assert feats_zero["c40_imag"] == 0.0
    assert feats_zero["c40_magnitude"] == 0.0

# 5. Test Autocorrelation Feature Family
def test_autocorrelation_features():
    # Test case 1: Constant signal x[n] = 1.0
    # R[k] = 1.0 for all lags, normalized R_norm[k] = 1.0
    x = np.ones(128, dtype=np.complex64)
    feats = extract_autocorrelation_features(x)
    
    for k in [1, 2, 4, 8, 16]:
        assert np.allclose(feats[f"autocorr_lag_{k}_real"], 1.0, atol=1e-5)
        assert np.allclose(feats[f"autocorr_lag_{k}_imag"], 0.0, atol=1e-5)
        assert np.allclose(feats[f"autocorr_lag_{k}_magnitude"], 1.0, atol=1e-5)
        
    # Test case 2: Impulse signal x[0] = 1.0, else 0
    # R[0] = 1/128, R[k] = 0 for k >= 1
    x_imp = np.zeros(128, dtype=np.complex64)
    x_imp[0] = 1.0
    feats_imp = extract_autocorrelation_features(x_imp)
    
    for k in [1, 2, 4, 8, 16]:
        assert np.allclose(feats_imp[f"autocorr_lag_{k}_real"], 0.0, atol=1e-5)
        assert np.allclose(feats_imp[f"autocorr_lag_{k}_imag"], 0.0, atol=1e-5)
        assert np.allclose(feats_imp[f"autocorr_lag_{k}_magnitude"], 0.0, atol=1e-5)

# 6. Test High-Level Extractor API
def test_high_level_extractor():
    # Create valid sample shape [2, 128]
    # Channels: 0 = I, 1 = Q
    samples = np.random.randn(2, 128).astype(np.float32)
    
    names, vec = extract_features(samples)
    
    assert len(names) == NUM_FEATURES
    assert len(vec) == NUM_FEATURES
    assert names == FEATURE_NAMES
    assert vec.dtype == np.float32
    assert np.isfinite(vec).all()
    
    # Test determinism
    _, vec_2 = extract_features(samples)
    assert np.array_equal(vec, vec_2)

# 7. Test Batch Feature Extractor
def test_batch_extractor():
    N = 10
    batch_samples = np.random.randn(N, 2, 128).astype(np.float32)
    
    X = extract_batch_features(batch_samples)
    assert X.shape == (N, NUM_FEATURES)
    assert X.dtype == np.float32
    assert np.isfinite(X).all()
    
    # Check alignment: extracting individually must equal the batch result
    for i in range(N):
        _, single_vec = extract_features(batch_samples[i])
        assert np.allclose(X[i], single_vec, atol=1e-5)

# 8. Test Feature Matrix Extraction with Mock Dataset
@pytest.fixture
def mock_loader_pickle():
    mock_data = {}
    snrs = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    examples_per_key = 2
    
    for mod in MODULATION_CLASSES:
        for snr in snrs:
            mock_data[(mod, snr)] = np.random.randn(examples_per_key, 2, 128).astype(np.float32)
            
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        pickle.dump(mock_data, tmp)
        tmp_path = tmp.name
        
    yield tmp_path
    
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

def test_feature_matrix_extraction(mock_loader_pickle):
    DatasetCache.clear()
    dataset = RadioMLDataset(pickle_path=mock_loader_pickle)
    
    X, y, snrs = extract_feature_matrix(dataset)
    
    assert X.shape == (len(dataset), NUM_FEATURES)
    assert y.shape == (len(dataset),)
    assert snrs.shape == (len(dataset),)
    
    assert X.dtype == np.float32
    assert y.dtype == np.int32
    assert snrs.dtype == np.int32
    
    # Verify labels and snrs match the dataset
    assert np.array_equal(y, dataset.class_indices)
    assert np.array_equal(snrs, dataset.snrs)
