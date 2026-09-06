import os
import torch
import pytest
import numpy as np

from ml.dataset.labels import MODULATION_CLASSES, get_class_index
from ml.cnn_model.architecture import RawIQCNN
from ml.synthetic.dataset import load_synthetic_dataset
from ml.synthetic.evaluate import get_state_dict_hash
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Test M5 Checkpoint Loading & Preprocessing Alignment
def test_cross_domain_checkpoint_loading():
    checkpoint_path = "models/m5_iq_cnn.pt"
    if not os.path.exists(checkpoint_path):
        pytest.skip(f"M5 model checkpoint not found at {checkpoint_path}. Skipping.")
        
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert "model_state_dict" in checkpoint
    assert "rms_factor" in checkpoint
    
    model = RawIQCNN(num_classes=11)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Assert model has correct number of classes in output layer
    assert model.fc[-1].out_features == 11
    assert float(checkpoint["rms_factor"]) > 0.0

# 2. Test Weight Freezing & Parameter Integrity Check
def test_cross_domain_weight_freezing():
    model = RawIQCNN(num_classes=11)
    model.eval()
    
    # Capture initial hash
    initial_hash = get_state_dict_hash(model.state_dict())
    
    # Run a forward pass
    x = torch.randn(4, 2, 128)
    _ = model(x)
    
    # Capture final hash and verify match
    final_hash = get_state_dict_hash(model.state_dict())
    assert initial_hash == final_hash

# 3. Test Prediction Shapes & Probability Constraints
def test_cross_domain_shapes_and_probabilities():
    model = RawIQCNN(num_classes=11)
    model.eval()
    
    x = torch.randn(5, 2, 128)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).numpy()
        preds = np.argmax(probs, axis=1)
        
    assert logits.shape == (5, 11)
    assert probs.shape == (5, 11)
    assert preds.shape == (5,)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)

# 4. Test 5-Class Evaluation Constraints
def test_five_class_evaluation_metrics():
    # True labels (BPSK, QPSK, 8PSK, QAM16, QAM64)
    supported_classes = ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64"]
    supported_indices = [get_class_index(cls) for cls in supported_classes]
    
    # 5 true examples from supported classes
    y_true = np.array([supported_indices[0], supported_indices[1], supported_indices[2], supported_indices[3], supported_indices[4]])
    
    # Predictions (one of which is an UNSUPPORTED class, e.g. AM-DSB class index 1)
    unsupported_idx = get_class_index("AM-DSB")
    assert unsupported_idx not in supported_indices
    
    y_pred = np.array([supported_indices[0], supported_indices[1], unsupported_idx, supported_indices[3], supported_indices[4]])
    
    # Compute accuracy
    acc = accuracy_score(y_true, y_pred)
    assert acc == 0.8 # 4 out of 5 correct
    
    # Compute macro precision/recall/F1 over ONLY the 5 supported classes
    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=supported_indices, average="macro", zero_division=0
    )
    
    # BPSK (supported_indices[0]): TP=1, FP=0, FN=0 -> F1=1.0
    # QPSK (supported_indices[1]): TP=1, FP=0, FN=0 -> F1=1.0
    # 8PSK (supported_indices[2]): TP=0, FP=0, FN=1 -> F1=0.0 (predicted as AM-DSB)
    # QAM16 (supported_indices[3]): TP=1, FP=0, FN=0 -> F1=1.0
    # QAM64 (supported_indices[4]): TP=1, FP=0, FN=0 -> F1=1.0
    # Average F1 should be (1+1+0+1+1)/5 = 0.8
    assert np.isclose(f1, 0.8)

# 5. Test Dataset Loading & Grouping Checks
def test_cross_domain_metadata_grouping():
    # Mock metadata list
    metadata = [
        {"experiment": "awgn", "sweep_parameter": "snr", "sweep_value": 10.0, "modulation": "BPSK", "class_index": get_class_index("BPSK"), "random_seed": 42},
        {"experiment": "awgn", "sweep_parameter": "snr", "sweep_value": -10.0, "modulation": "QPSK", "class_index": get_class_index("QPSK"), "random_seed": 43},
        {"experiment": "frequency_offset", "sweep_parameter": "frequency_offset", "sweep_value": 500.0, "modulation": "BPSK", "class_index": get_class_index("BPSK"), "random_seed": 44},
    ]
    
    experiments = np.array([m["experiment"] for m in metadata])
    sweep_values = np.array([m["sweep_value"] for m in metadata])
    
    # AWGN masks
    awgn_mask = (experiments == "awgn")
    assert np.sum(awgn_mask) == 2
    
    # Frequency offset masks
    freq_mask = (experiments == "frequency_offset")
    assert np.sum(freq_mask) == 1
    assert sweep_values[freq_mask][0] == 500.0
