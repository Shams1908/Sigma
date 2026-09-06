import os
import tempfile
import json
import joblib
import pytest
import numpy as np

from ml.dataset.loader import RadioMLDataset, DatasetCache
from ml.dataset.labels import MODULATION_CLASSES, get_class_index, get_modulation_name
from ml.features.schema import FEATURE_NAMES, NUM_FEATURES
from ml.baselines.inference import predict, get_baseline_model, clear_model_cache

from sklearn.ensemble import RandomForestClassifier

# 1. Verification of Feature Matrix and Split Indices
def test_feature_matrix_loading():
    npz_path = "datasets/processed/RML2016.10a_features.npz"
    if not os.path.exists(npz_path):
        pytest.skip(f"Pre-extracted feature file not found at {npz_path}. Skipping.")
        
    data = np.load(npz_path)
    assert "X" in data
    assert "y" in data
    assert "snrs" in data
    assert "feature_names" in data
    
    X = data["X"]
    y = data["y"]
    snrs = data["snrs"]
    feature_names = data["feature_names"]
    
    assert X.shape == (220000, 36)
    assert y.shape == (220000,)
    assert snrs.shape == (220000,)
    assert len(feature_names) == 36
    assert list(feature_names) == FEATURE_NAMES
    
    # Check no NaNs/Infs
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()

def test_split_indices_and_overlap():
    train_ds, val_ds, test_ds = RadioMLDataset.get_splits()
    
    assert len(train_ds) == 154000
    assert len(val_ds) == 33000
    assert len(test_ds) == 33000
    
    train_indices = set(train_ds.indices)
    val_indices = set(val_ds.indices)
    test_indices = set(test_ds.indices)
    
    # Assert zero overlap
    assert len(train_indices.intersection(val_indices)) == 0
    assert len(train_indices.intersection(test_indices)) == 0
    assert len(val_indices.intersection(test_indices)) == 0

def test_leakage_audit_assertions():
    npz_path = "datasets/processed/RML2016.10a_features.npz"
    if not os.path.exists(npz_path):
        pytest.skip("Features file missing. Skipping.")
        
    data = np.load(npz_path)
    X = data["X"]
    y = data["y"]
    snrs = data["snrs"]
    
    # Verify no accidental label or SNR column leakage in features
    for c in range(NUM_FEATURES):
        column = X[:, c]
        # Exact column matches
        assert not np.array_equal(column, y), f"Leakage: column {c} is identical to labels y!"
        assert not np.array_equal(column, snrs), f"Leakage: column {c} is identical to snrs!"

# 2. Deterministic Model Training & Predictions
def test_deterministic_rf_training():
    # Setup small mock dataset
    np.random.seed(42)
    X_mock = np.random.randn(100, NUM_FEATURES).astype(np.float32)
    y_mock = np.random.randint(0, 11, size=100)
    
    # Train deterministic RF
    rf1 = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    rf1.fit(X_mock, y_mock)
    
    rf2 = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    rf2.fit(X_mock, y_mock)
    
    # Predict on test data
    X_test = np.random.randn(20, NUM_FEATURES).astype(np.float32)
    
    preds1 = rf1.predict(X_test)
    preds2 = rf2.predict(X_test)
    
    probs1 = rf1.predict_proba(X_test)
    probs2 = rf2.predict_proba(X_test)
    
    assert np.array_equal(preds1, preds2)
    assert np.allclose(probs1, probs2)
    
    # Test prediction shapes
    assert preds1.shape == (20,)
    assert probs1.shape == (20, 11)
    
    # Verify probabilities sum to approximately 1
    assert np.allclose(np.sum(probs1, axis=1), 1.0, atol=1e-5)

# 3. Model Serialization and Reloading
def test_serialization_and_reloading():
    np.random.seed(42)
    X_mock = np.random.randn(100, NUM_FEATURES).astype(np.float32)
    y_mock = np.random.randint(0, 11, size=100)
    
    # Train a model
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X_mock, y_mock)
    
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        joblib.dump(model, tmp_path)
        
        # Reload
        reloaded = joblib.load(tmp_path)
        
        # Test predictions are identical
        X_test = np.random.randn(20, NUM_FEATURES).astype(np.float32)
        preds_orig = model.predict(X_test)
        preds_reloaded = reloaded.predict(X_test)
        
        assert np.array_equal(preds_orig, preds_reloaded)
        
        probs_orig = model.predict_proba(X_test)
        probs_reloaded = reloaded.predict_proba(X_test)
        assert np.allclose(probs_orig, probs_reloaded)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# 4. Inference API Integration
def test_inference_predict_api():
    # Skip if model hasn't been serialized yet
    if not (os.path.exists("models/baseline_rf.joblib") or os.path.exists("models/baseline_hgb.joblib")):
        pytest.skip("No baseline model serialized yet. Skipping API tests.")
        
    clear_model_cache()
    
    # 4a. Single sample inference
    single_sample = np.random.randn(NUM_FEATURES).astype(np.float32)
    res_single = predict(single_sample)
    
    assert isinstance(res_single, dict)
    assert "class_index" in res_single
    assert "class_name" in res_single
    assert "probabilities" in res_single
    assert "confidence" in res_single
    
    assert isinstance(res_single["class_index"], int)
    assert isinstance(res_single["class_name"], str)
    assert res_single["class_name"] in MODULATION_CLASSES
    assert res_single["probabilities"].shape == (11,)
    assert isinstance(res_single["confidence"], float)
    assert np.allclose(np.sum(res_single["probabilities"]), 1.0, atol=1e-5)
    
    # 4b. Batch inference
    batch_size = 5
    batch_samples = np.random.randn(batch_size, NUM_FEATURES).astype(np.float32)
    res_batch = predict(batch_samples)
    
    assert isinstance(res_batch, dict)
    assert res_batch["class_index"].shape == (batch_size,)
    assert res_batch["class_name"].shape == (batch_size,)
    assert res_batch["probabilities"].shape == (batch_size, 11)
    assert res_batch["confidence"].shape == (batch_size,)
    
    # Ensure correct types
    assert res_batch["class_index"].dtype == np.int32
    assert res_batch["probabilities"].dtype == np.float32
    assert res_batch["confidence"].dtype == np.float32
    
    # Check probability summing
    assert np.allclose(np.sum(res_batch["probabilities"], axis=1), 1.0, atol=1e-5)
    
    # 4c. Check error cases
    # NaNs/Infs checking
    corrupted_sample = single_sample.copy()
    corrupted_sample[0] = np.nan
    with pytest.raises(ValueError):
        predict(corrupted_sample)
        
    # Shape checking
    wrong_shape = np.random.randn(NUM_FEATURES - 1).astype(np.float32)
    with pytest.raises(ValueError):
        predict(wrong_shape)
