import os
import torch
import pytest
import numpy as np

from ml.dataset.labels import MODULATION_CLASSES
from ml.cnn_model.architecture import RawIQCNN
from ml.cnn_model.dataset import RadioMLPyTorchDataset
from ml.cnn_model.inference import predict_iq, get_cnn_model, clear_cnn_cache
from ml.dataset.loader import RadioMLDataset

# 1. Model Structure and Forward Pass Tests
def test_cnn_model_construction():
    model = RawIQCNN(num_classes=11)
    
    # Check that model structure contains the expected blocks
    assert hasattr(model, "block1")
    assert hasattr(model, "block2")
    assert hasattr(model, "block3")
    assert hasattr(model, "fc")
    
    # Check shape of forward pass
    # Input batch size = 4, channels = 2, time samples = 128
    batch_size = 4
    x = torch.randn(batch_size, 2, 128)
    out = model(x)
    
    assert out.shape == (batch_size, 11)
    # Check no NaN/Infs in logits
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

# 2. Probability Constraints Tests
def test_cnn_probabilities():
    model = RawIQCNN(num_classes=11)
    model.eval()
    
    x = torch.randn(2, 2, 128)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).numpy()
        
    assert probs.shape == (2, 11)
    # Probability rows sum approximately to 1
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)
    # Check range is [0, 1]
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)

# 3. Checkpoint Save/Load and Predictions Test
def test_checkpoint_save_and_load():
    model = RawIQCNN(num_classes=11)
    model.eval()
    
    # Save a temporary checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "rms_factor": 0.006,
        "val_macro_f1": 0.55
    }
    
    temp_ckpt_path = "models/temp_test_ckpt.pt"
    os.makedirs("models", exist_ok=True)
    
    try:
        torch.save(checkpoint, temp_ckpt_path)
        
        # Load model using PyTorch
        reloaded_ckpt = torch.load(temp_ckpt_path, map_location="cpu")
        reloaded_model = RawIQCNN(num_classes=11)
        reloaded_model.load_state_dict(reloaded_ckpt["model_state_dict"])
        reloaded_model.eval()
        
        # Verify predictions on mock sample are identical
        x = torch.randn(3, 2, 128)
        with torch.no_grad():
            out_orig = model(x).numpy()
            out_reloaded = reloaded_model(x).numpy()
            
        assert np.allclose(out_orig, out_reloaded, atol=1e-5)
    finally:
        if os.path.exists(temp_ckpt_path):
            os.remove(temp_ckpt_path)

# 4. Inference API Integration Tests
def test_cnn_inference_predict_api():
    # Skip if model hasn't been trained and serialized yet
    model_path = "models/m5_iq_cnn.pt"
    if not os.path.exists(model_path):
        pytest.skip(f"Champion CNN checkpoint not found at {model_path}. Skipping.")
        
    clear_cnn_cache()
    
    # 4a. Single sample inference
    single_sample = np.random.randn(2, 128).astype(np.float32)
    res = predict_iq(single_sample, model_path=model_path)
    
    assert isinstance(res, dict)
    assert "class_index" in res
    assert "class_name" in res
    assert "probabilities" in res
    assert "confidence" in res
    
    assert isinstance(res["class_index"], int)
    assert isinstance(res["class_name"], str)
    assert res["class_name"] in MODULATION_CLASSES
    assert res["probabilities"].shape == (11,)
    assert isinstance(res["confidence"], float)
    assert np.allclose(np.sum(res["probabilities"]), 1.0, atol=1e-5)
    
    # 4b. Batch inference
    batch_size = 3
    batch_samples = np.random.randn(batch_size, 2, 128).astype(np.float32)
    res_batch = predict_iq(batch_samples, model_path=model_path)
    
    assert isinstance(res_batch, dict)
    assert res_batch["class_index"].shape == (batch_size,)
    assert res_batch["class_name"].shape == (batch_size,)
    assert res_batch["probabilities"].shape == (batch_size, 11)
    assert res_batch["confidence"].shape == (batch_size,)
    
    assert res_batch["class_index"].dtype == np.int32
    assert res_batch["probabilities"].dtype == np.float32
    assert res_batch["confidence"].dtype == np.float32
    
    # 4c. Check error handling
    # NaN/Inf check
    nan_sample = single_sample.copy()
    nan_sample[0, 0] = np.nan
    with pytest.raises(ValueError):
        predict_iq(nan_sample, model_path=model_path)
        
    # Shape check
    wrong_shape = np.random.randn(3, 128).astype(np.float32)
    with pytest.raises(ValueError):
        predict_iq(wrong_shape, model_path=model_path)

# 5. Dataset Split Verification
def test_dataset_split_integrity():
    train_ds, val_ds, test_ds = RadioMLDataset.get_splits()
    
    train_indices = set(train_ds.indices)
    val_indices = set(val_ds.indices)
    test_indices = set(test_ds.indices)
    
    assert len(train_indices.intersection(val_indices)) == 0
    assert len(train_indices.intersection(test_indices)) == 0
    assert len(val_indices.intersection(test_indices)) == 0
