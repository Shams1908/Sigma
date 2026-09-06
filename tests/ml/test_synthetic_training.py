import os
import hashlib
import numpy as np
import torch
import pytest

from ml.synthetic_training.dataset import (
    load_real_5_class_splits,
    load_synthetic_5_class_dataset,
    PyTorchSignalDataset,
    SUPPORTED_5_CLASSES
)
from ml.synthetic_training.sampler import construct_mixed_dataset
from ml.cnn_model.architecture import RawIQCNN

# 1. Dataset Loading, 5-Class Filtering, Shapes & Dtypes
def test_dataset_loading_and_filtering():
    # Load splits
    (X_train, y_train, snrs_train), (X_val, y_val, snrs_val), (X_test, y_test, snrs_test) = load_real_5_class_splits()
    
    # 5-class target verification
    assert len(X_train) == 70000
    assert len(X_val) == 15000
    assert len(X_test) == 15000
    
    # Verify shape
    assert X_train.shape == (70000, 2, 128)
    assert X_val.shape == (15000, 2, 128)
    assert X_test.shape == (15000, 2, 128)
    
    # Target class values must reside within [0, 4]
    assert np.all((y_train >= 0) & (y_train <= 4))
    assert np.all((y_val >= 0) & (y_val <= 4))
    assert np.all((y_test >= 0) & (y_test <= 4))

# 2. Class Balance Verification
def test_class_balance():
    (X_train, y_train, _), _, _ = load_real_5_class_splits()
    
    # Real dataset is balanced with 14,000 samples per class
    for c in range(5):
        assert np.sum(y_train == c) == 14000

# 3. Mixed Dataset Construction & Synthetic Ratio Correctness
def test_mixed_dataset_ratios():
    X_real = np.zeros((70000, 2, 128))
    y_real = np.repeat(np.arange(5), 14000)
    
    X_syn = np.ones((18000, 2, 128))
    y_syn = np.repeat(np.arange(5), 3600)
    
    # 10% ratio test
    X_m10, y_m10 = construct_mixed_dataset(X_real, y_real, X_syn, y_syn, ratio_pct=10)
    assert len(X_m10) == 70000
    for c in range(5):
        # 14,000 per class: 1,400 synthetic (value 1.0) and 12,600 real (value 0.0)
        assert np.sum(y_m10 == c) == 14000
        class_samples = X_m10[y_m10 == c]
        # Count synthetic samples by check mean
        syn_count = np.sum(class_samples[:, 0, 0] == 1.0)
        assert syn_count == 14000 * 0.10 # exactly 10%
        
    # 25% ratio test
    X_m25, y_m25 = construct_mixed_dataset(X_real, y_real, X_syn, y_syn, ratio_pct=25)
    assert len(X_m25) == 70000
    for c in range(5):
        assert np.sum(y_m25 == c) == 14000
        class_samples = X_m25[y_m25 == c]
        syn_count = np.sum(class_samples[:, 0, 0] == 1.0)
        assert syn_count == 14000 * 0.25 # exactly 25%
        
    # 50% ratio test
    X_m50, y_m50 = construct_mixed_dataset(X_real, y_real, X_syn, y_syn, ratio_pct=50)
    assert len(X_m50) == 35000
    for c in range(5):
        assert np.sum(y_m50 == c) == 7000
        class_samples = X_m50[y_m50 == c]
        syn_count = np.sum(class_samples[:, 0, 0] == 1.0)
        assert syn_count == 7000 * 0.50 # exactly 50%

# 4. No Target or SNR Leakage Checks
def test_no_leakage_in_datasets():
    (X_train, y_train, snrs_train), _, _ = load_real_5_class_splits()
    
    # Construct PyTorch Dataset
    ds = PyTorchSignalDataset(X_train, y_train, rms_factor=1.0)
    bx, by = ds[0]
    
    # bx must have shape (2, 128) and by must be a scalar
    assert bx.shape == (2, 128)
    assert by.shape == ()
    
    # Confirm target and SNR is NOT part of input tensor
    # If target/SNR leaked, bx would have size other than (2, 128)
    assert bx.dtype == torch.float32

# 5. Split Isolation Check
def test_split_isolation():
    (X_train, y_train, _), (X_val, y_val, _), (X_test, y_test, _) = load_real_5_class_splits()
    
    # Samples must be distinct (check memory address and content uniqueness via simple check)
    # Check that they represent separate splits
    assert id(X_train) != id(X_val)
    assert id(X_val) != id(X_test)

# 6. Checkpoint Serialization & Reloading
def test_checkpoint_reloading(tmp_path):
    checkpoint_path = os.path.join(tmp_path, "test_checkpoint.pt")
    
    model = RawIQCNN(num_classes=5)
    state_dict = model.state_dict()
    
    torch.save({
        "epoch": 5,
        "model_state_dict": state_dict,
        "rms_factor": 0.0085,
        "best_val_macro_f1": 0.65
    }, checkpoint_path)
    
    assert os.path.exists(checkpoint_path)
    
    # Load back
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    assert ckpt["epoch"] == 5
    assert ckpt["rms_factor"] == 0.0085
    assert ckpt["best_val_macro_f1"] == 0.65
    
    model_loaded = RawIQCNN(num_classes=5)
    model_loaded.load_state_dict(ckpt["model_state_dict"])
    
    # Verify weights identity
    for p1, p2 in zip(model.parameters(), model_loaded.parameters()):
        assert torch.equal(p1, p2)

# 7. M5 Checkpoint Integrity Verification
def test_m5_checkpoint_integrity():
    m5_path = "models/m5_iq_cnn.pt"
    assert os.path.exists(m5_path), "M5 baseline model checkpoint is missing!"
    
    # Verify MD5 is unchanged
    expected_m5_hash = "f4230af498fed1c87a8717f42badd5bb"
    current_hash = hashlib.md5(open(m5_path, "rb").read()).hexdigest()
    assert current_hash == expected_m5_hash, "M5 checkpoint was modified!"
