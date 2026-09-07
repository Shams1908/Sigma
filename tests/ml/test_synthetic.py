import os
import shutil
import tempfile
import numpy as np
import pytest

from ml.dataset.labels import INDEX_TO_MODULATION, MODULATION_CLASSES, get_class_index
from ml.synthetic.config import (
    SUPPORTED_MODULATION_CLASSES,
    NOMINAL_CONFIG,
    SWEEP_EXPERIMENTS
)
from ml.synthetic.generator import generate_synthetic_evaluation_dataset
from ml.synthetic.dataset import load_synthetic_dataset, validate_synthetic_dataset

@pytest.fixture
def temp_dir():
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)

# 1. Test Dataset Generation, Shape, Dtype, and Save/Load
def test_synthetic_dataset_generation_and_load(temp_dir):
    # Use 1 example per combination to keep it fast
    num_ex = 1
    X, y, metadata = generate_synthetic_evaluation_dataset(
        num_examples_per_cond=num_ex,
        output_dir=temp_dir
    )
    
    # Total conditions = 5 (AWGN) + 5 (Freq) + 5 (Phase) + 5 (Amp) + 5 (Phase imb) + 5 (DC) + 6 (Timing) = 36 conditions
    # Total examples = 36 conditions * 5 modulations * 1 ex = 180 examples
    expected_count = 180
    assert X.shape == (expected_count, 2, 128)
    assert X.dtype == np.float32
    assert y.shape == (expected_count,)
    assert y.dtype == np.int32
    assert len(metadata) == expected_count
    
    # Check loaded files match generated ones
    X_load, y_load, metadata_load = load_synthetic_dataset(data_dir=temp_dir)
    assert np.array_equal(X, X_load)
    assert np.array_equal(y, y_load)
    assert len(metadata_load) == len(metadata)
    assert metadata_load[0]["modulation"] == metadata[0]["modulation"]

# 2. Test Determinism (Same inputs + seed -> identical samples)
def test_synthetic_determinism(temp_dir):
    dir1 = os.path.join(temp_dir, "run1")
    dir2 = os.path.join(temp_dir, "run2")
    
    X1, y1, meta1 = generate_synthetic_evaluation_dataset(num_examples_per_cond=1, output_dir=dir1)
    X2, y2, meta2 = generate_synthetic_evaluation_dataset(num_examples_per_cond=1, output_dir=dir2)
    
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)
    assert len(meta1) == len(meta2)
    for m1, m2 in zip(meta1, meta2):
        assert m1["modulation"] == m2["modulation"]
        assert m1["sweep_value"] == m2["sweep_value"]
        assert m1["random_seed"] == m2["random_seed"]

# 3. Test Validation Logic
def test_synthetic_validation():
    # Valid structures
    X = np.random.randn(5, 2, 128).astype(np.float32)
    
    metadata = []
    y_list = []
    for i in range(5):
        mod = SUPPORTED_MODULATION_CLASSES[i % len(SUPPORTED_MODULATION_CLASSES)]
        c_idx = get_class_index(mod)
        y_list.append(c_idx)
        metadata.append({
            "index": i,
            "experiment": "awgn",
            "sweep_parameter": "snr",
            "sweep_value": 10.0,
            "modulation": mod,
            "class_index": c_idx,
            "random_seed": 42
        })
    y = np.array(y_list, dtype=np.int32)
        
    # Should pass without error
    validate_synthetic_dataset(X, y, metadata)
    
    # 3a. Invalid shape rejection
    X_bad = np.random.randn(5, 3, 128).astype(np.float32)
    with pytest.raises(ValueError):
        validate_synthetic_dataset(X_bad, y, metadata)
        
    # 3b. Invalid type rejection
    X_bad_type = X.astype(np.float64)
    with pytest.raises(TypeError):
        validate_synthetic_dataset(X_bad_type, y, metadata)
        
    # 3c. NaNs/Infs rejection
    X_nan = X.copy()
    X_nan[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_synthetic_dataset(X_nan, y, metadata)
        
    # 3d. Label alignment mismatch
    y_mismatch = y.copy()
    y_mismatch[0] = 9
    with pytest.raises(ValueError):
        validate_synthetic_dataset(X, y_mismatch, metadata)

# 4. Test One-at-a-Time Impairment Isolation
def test_impairment_isolation(temp_dir):
    X, y, metadata = generate_synthetic_evaluation_dataset(
        num_examples_per_cond=1,
        output_dir=temp_dir
    )
    
    # Loop through and assert that for each experiment, only its specific sweep parameter
    # is active/different from nominal, while all other parameters are held constant.
    for entry in metadata:
        exp = entry["experiment"]
        param = entry["sweep_parameter"]
        val = entry["sweep_value"]
        cfg = entry["generation_config"]
        
        # Check standard nominal waveform values are unchanged
        assert cfg["num_symbols"] == NOMINAL_CONFIG["num_symbols"]
        assert cfg["sample_rate"] == NOMINAL_CONFIG["sample_rate"]
        assert cfg["symbol_rate"] == NOMINAL_CONFIG["symbol_rate"]
        assert cfg["samples_per_symbol"] == NOMINAL_CONFIG["samples_per_symbol"]
        assert cfg["filter_span_symbols"] == NOMINAL_CONFIG["filter_span_symbols"]
        assert cfg["rolloff"] == NOMINAL_CONFIG["rolloff"]
        
        # Check impairment isolation:
        # For non-active sweep parameter names, verify they remain None or default (0.0).
        if exp == "awgn":
            # Only SNR changes, other impairments must be None
            assert cfg["snr"] == val
            assert cfg["frequency_offset"] is None
            assert cfg["phase_offset"] is None
            assert cfg["timing_offset"] is None
            assert cfg["dc_offset_i"] is None
            assert cfg["dc_offset_q"] is None
            assert cfg["iq_amplitude_imbalance"] is None
            assert cfg["iq_phase_imbalance"] is None
            
        elif exp == "frequency_offset":
            assert cfg["frequency_offset"] == val
            assert cfg["snr"] == 18.0
            assert cfg["phase_offset"] is None
            assert cfg["timing_offset"] is None
            assert cfg["dc_offset_i"] is None
            assert cfg["dc_offset_q"] is None
            assert cfg["iq_amplitude_imbalance"] is None
            assert cfg["iq_phase_imbalance"] is None
            
        elif exp == "phase_offset":
            assert cfg["phase_offset"] == val
            assert cfg["snr"] == 18.0
            assert cfg["frequency_offset"] is None
            assert cfg["timing_offset"] is None
            assert cfg["dc_offset_i"] is None
            assert cfg["dc_offset_q"] is None
            assert cfg["iq_amplitude_imbalance"] is None
            assert cfg["iq_phase_imbalance"] is None
            
        elif exp == "iq_amplitude_imbalance":
            assert cfg["iq_amplitude_imbalance"] == val
            assert cfg["snr"] == 18.0
            assert cfg["frequency_offset"] is None
            assert cfg["phase_offset"] is None
            assert cfg["timing_offset"] is None
            assert cfg["dc_offset_i"] is None
            assert cfg["dc_offset_q"] is None
            assert cfg["iq_phase_imbalance"] is None
            
        elif exp == "iq_phase_imbalance":
            assert cfg["iq_phase_imbalance"] == val
            assert cfg["snr"] == 18.0
            assert cfg["frequency_offset"] is None
            assert cfg["phase_offset"] is None
            assert cfg["timing_offset"] is None
            assert cfg["dc_offset_i"] is None
            assert cfg["dc_offset_q"] is None
            assert cfg["iq_amplitude_imbalance"] is None
            
        elif exp == "dc_offset":
            assert cfg["dc_offset_i"] == val
            assert cfg["dc_offset_q"] == val
            assert cfg["snr"] == 18.0
            assert cfg["frequency_offset"] is None
            assert cfg["phase_offset"] is None
            assert cfg["timing_offset"] is None
            assert cfg["iq_amplitude_imbalance"] is None
            assert cfg["iq_phase_imbalance"] is None
            
        elif exp == "timing_offset":
            assert cfg["timing_offset"] == val
            assert cfg["snr"] == 18.0
            assert cfg["frequency_offset"] is None
            assert cfg["phase_offset"] is None
            assert cfg["dc_offset_i"] is None
            assert cfg["dc_offset_q"] is None
            assert cfg["iq_amplitude_imbalance"] is None
            assert cfg["iq_phase_imbalance"] is None
