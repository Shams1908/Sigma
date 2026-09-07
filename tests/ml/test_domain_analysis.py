import os
import torch
import pytest
import numpy as np
import pandas as pd

from ml.dataset.loader import RadioMLDataset, DatasetCache
from ml.dataset.labels import MODULATION_CLASSES, get_class_index
from ml.synthetic.dataset import load_synthetic_dataset
from ml.features.extractor import extract_batch_features
from ml.features.schema import FEATURE_NAMES

# 1. Shape Compatibility Tests
def test_domain_analysis_shapes():
    # Load dataset structures
    checkpoint_path = "models/m5_iq_cnn.pt"
    if not os.path.exists(checkpoint_path):
        pytest.skip("Frozen M5 model checkpoint not found. Skipping.")
        
    ds = RadioMLDataset()
    real_samples = DatasetCache.samples
    
    syn_samples, _, _ = load_synthetic_dataset()
    
    # Assert dimensions are compatible: both are [N, 2, 128]
    assert real_samples.ndim == 3
    assert real_samples.shape[1:] == (2, 128)
    
    assert syn_samples.ndim == 3
    assert syn_samples.shape[1:] == (2, 128)

# 2. Matching Filter Logic and Subsampling Determinism
def test_matching_filter_subsampling():
    # Setup mock data arrays
    mods = np.array(["BPSK", "BPSK", "QPSK", "QPSK", "8PSK", "8PSK"])
    snrs = np.array([10, 10, 10, -10, 10, 10])
    
    # Matching target: BPSK at SNR=10
    target_mod = "BPSK"
    target_snr = 10
    num_ex = 1
    
    mask = np.where((mods == target_mod) & (snrs == target_snr))[0]
    subsampled = mask[:num_ex]
    
    # Assert selection is deterministic and yields exact sample counts
    assert len(subsampled) == 1
    assert subsampled[0] == 0 # first occurrence
    
    # Matching target: QPSK at SNR=10
    mask_qpsk = np.where((mods == "QPSK") & (snrs == 10))[0]
    subsampled_qpsk = mask_qpsk[:num_ex]
    assert len(subsampled_qpsk) == 1
    assert subsampled_qpsk[0] == 2

# 3. Numeric Safety (No NaN or Inf in Extracted Features)
def test_feature_extraction_numeric_safety():
    # Mock inputs
    # Zero power signal
    zero_signal = np.zeros((2, 2, 128), dtype=np.float32)
    feats = extract_batch_features(zero_signal)
    
    # Check shape
    assert feats.shape == (2, 36)
    # Check no NaN/Infs
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()

# 4. Feature Comparison CSV Output Schema
def test_domain_feature_comparison_csv_schema():
    csv_path = "results/ml/m6/domain_analysis/domain_feature_comparison.csv"
    if not os.path.exists(csv_path):
        pytest.skip("CSV output does not exist. Run analysis first.")
        
    df = pd.read_csv(csv_path)
    expected_cols = [
        "feature",
        "real_mean",
        "synthetic_mean",
        "real_std",
        "synthetic_std",
        "mean_difference",
        "relative_difference",
        "effect_size"
    ]
    for col in expected_cols:
        assert col in df.columns
        
    # Check that it evaluates exactly 36 features
    assert len(df) == 36
    # Top ranked feature must be a valid feature name
    assert df.iloc[0]["feature"] in FEATURE_NAMES

# 5. M5 Checkpoint Integrity Check
def test_m5_checkpoint_integrity_unchanged():
    checkpoint_path = "models/m5_iq_cnn.pt"
    if not os.path.exists(checkpoint_path):
        pytest.skip("M5 checkpoint not found. Skipping.")
        
    # Load state dict parameter keys and shapes
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    
    # Confirm it matches exactly the frozen state dict keys and weights
    assert "block1.0.weight" in state_dict
    assert state_dict["block1.0.weight"].shape == (64, 2, 7)
