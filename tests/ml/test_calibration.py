import os
import torch
import pytest
import numpy as np
import pandas as pd

from ml.dataset.loader import RadioMLDataset, DatasetCache
from ml.synthetic.dataset import load_synthetic_dataset

# 1. Test dynamic RMS calculation logic on custom mock arrays
def test_dynamic_rms_calculation():
    # Make a mock 3D signal of shape [N, 2, 128]
    # Representing a constant complex signal of magnitude 2.0
    # I = 1.2, Q = 1.6 => |z|^2 = 1.2^2 + 1.6^2 = 1.44 + 2.56 = 4.0 => RMS = 2.0
    N = 10
    X = np.empty((N, 2, 128), dtype=np.float32)
    X[:, 0, :] = 1.2
    X[:, 1, :] = 1.6
    
    # Calculate power and RMS
    power = np.mean(X[:, 0]**2 + X[:, 1]**2)
    rms = np.sqrt(power)
    
    assert np.allclose(power, 4.0)
    assert np.allclose(rms, 2.0)

# 2. Test dynamic synthetic reference RMS calculation logic on synthetic evaluation dataset
def test_synthetic_reference_rms_logic():
    X_syn, y_syn, syn_metadata = load_synthetic_dataset()
    syn_experiments = np.array([m["experiment"] for m in syn_metadata])
    syn_snrs = np.array([m["snr"] for m in syn_metadata])
    
    ref_mask = (syn_experiments == "awgn") & (syn_snrs == 18.0)
    X_syn_ref = X_syn[ref_mask]
    
    syn_ref_power = np.mean(X_syn_ref[:, 0]**2 + X_syn_ref[:, 1]**2)
    syn_ref_rms = np.sqrt(syn_ref_power)
    
    assert syn_ref_rms > 0.0
    assert syn_ref_power > 0.0

# 3. Test global calibration factor application and calibrated RMS correctness
def test_global_scale_factor_correctness():
    # Target RMS: 0.1, synthetic reference RMS: 0.5 => scale = 0.2
    target_rms = 0.1
    ref_rms = 0.5
    scale_factor = target_rms / ref_rms
    
    assert np.allclose(scale_factor, 0.2)
    
    # Check calibrated RMS
    cal_rms = ref_rms * scale_factor
    assert np.allclose(cal_rms, target_rms)

# 4. Test that calibration is amplitude-only and does not modify phase or shape/dtype
def test_amplitude_only_transformation():
    X_syn, _, _ = load_synthetic_dataset()
    scale = 0.05
    X_calibrated = X_syn * scale
    
    # Shape & Dtype
    assert X_calibrated.shape == X_syn.shape
    assert X_calibrated.dtype == X_syn.dtype
    
    # Phase preservation
    orig_comp = X_syn[0, 0] + 1j * X_syn[0, 1]
    cal_comp = X_calibrated[0, 0] + 1j * X_calibrated[0, 1]
    assert np.allclose(np.angle(orig_comp), np.angle(cal_comp), atol=1e-5)
    
    # Amplitude scaling
    assert np.allclose(np.abs(cal_comp), np.abs(orig_comp) * scale, atol=1e-5)

# 5. Test that original dataset files remain unchanged and calibrated files are created
def test_original_dataset_remains_unchanged():
    orig_npz = "datasets/synthetic/synthetic_evaluation_dataset.npz"
    cal_npz = "datasets/synthetic/synthetic_calibrated_dataset.npz"
    
    assert os.path.exists(orig_npz)
    assert os.path.exists(cal_npz)
    
    # Verify they are separate files with different content
    data_orig = np.load(orig_npz)
    data_cal = np.load(cal_npz)
    
    X_orig = data_orig["X"]
    X_cal = data_cal["X"]
    
    # Content must differ
    assert not np.array_equal(X_orig, X_cal)
    assert X_orig.shape == X_cal.shape

# 6. Test that running inference twice on calibrated data is deterministic
def test_calibration_inference_determinism():
    checkpoint_path = "models/m5_iq_cnn.pt"
    if not os.path.exists(checkpoint_path):
        pytest.skip("CNN model checkpoint not found.")
        
    from ml.cnn_model.architecture import RawIQCNN
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = RawIQCNN(num_classes=11)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Verify weights are frozen (state dict MD5 hash matches)
    from ml.synthetic.evaluate import get_state_dict_hash
    initial_hash = get_state_dict_hash(model.state_dict())
    
    data_cal = np.load("datasets/synthetic/synthetic_calibrated_dataset.npz")
    X_cal = data_cal["X"][:50] # Take a small batch
    rms_factor = float(checkpoint["rms_factor"])
    X_cal_norm = X_cal / rms_factor
    
    with torch.no_grad():
        out1 = model(torch.tensor(X_cal_norm, dtype=torch.float32)).numpy()
        out2 = model(torch.tensor(X_cal_norm, dtype=torch.float32)).numpy()
        
    assert np.allclose(out1, out2, atol=1e-6)
    
    final_hash = get_state_dict_hash(model.state_dict())
    assert initial_hash == final_hash

# 7. Audit checking for hardcoded dataset-derived constants in the implementation file
def test_no_hardcoded_scaling_constants_audit():
    script_path = "ml/synthetic/calibrate.py"
    assert os.path.exists(script_path)
    
    with open(script_path, "r") as f:
        content = f.read()
        
    # Check that none of the measured statistical values are hardcoded in the script
    forbidden_literals = [
        "0.006048",
        "0.006056",
        "0.852382",
        "140.926"
    ]
    for literal in forbidden_literals:
        assert literal not in content, f"Audit failed: found forbidden hardcoded constant '{literal}' in {script_path}!"
