import os
import json
import joblib
import numpy as np
from typing import Dict, Any, Union, Tuple, List

from ml.dataset.labels import INDEX_TO_MODULATION, MODULATION_CLASSES
from ml.features.schema import FEATURE_NAMES, NUM_FEATURES

# Cache to store loaded model and metadata in-memory
_MODEL_CACHE: Dict[str, Any] = {}

def get_baseline_model(model_dir: str = "models") -> Tuple[Any, Dict[str, Any]]:
    """
    Loads and caches the baseline model and its associated metadata.
    
    Args:
        model_dir (str): Directory containing serialized models.
        
    Returns:
        Tuple[Any, Dict]: The loaded scikit-learn model and metadata dictionary.
    """
    if "model" in _MODEL_CACHE:
        return _MODEL_CACHE["model"], _MODEL_CACHE["metadata"]
        
    # Search for available champion models in order of preference (RF, HGB)
    model_types = ["baseline_rf", "baseline_hgb"]
    selected_model_path = None
    selected_metadata_path = None
    
    for mtype in model_types:
        mpath = os.path.join(model_dir, f"{mtype}.joblib")
        metapath = os.path.join(model_dir, f"{mtype}_metadata.json")
        if os.path.exists(mpath) and os.path.exists(metapath):
            selected_model_path = mpath
            selected_metadata_path = metapath
            break
            
    if selected_model_path is None:
        raise FileNotFoundError(
            f"No trained baseline champion models found in '{model_dir}'. "
            f"Please run the training pipeline script `ml/baselines/train.py` first."
        )
        
    print(f"Loading cached baseline model from: {selected_model_path}")
    model = joblib.load(selected_model_path)
    
    with open(selected_metadata_path, "r") as f:
        metadata = json.load(f)
        
    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["metadata"] = metadata
    return model, metadata

def predict(features: np.ndarray, model_dir: str = "models") -> Dict[str, Any]:
    """
    Executes prediction using the trained classical baseline model.
    Supports both single sample (shape [36]) and batch input (shape [N, 36]).
    
    Args:
        features (np.ndarray): NumPy array containing 36 DSP features.
        model_dir (str): Directory containing serialized models.
        
    Returns:
        Dict[str, Any]: Dictionary containing:
            - class_index: Predicted integer class index (int or array)
            - class_name: Predicted canonical modulation name (str or array)
            - probabilities: Predicted class probability vector (array of shape [11] or [N, 11])
            - confidence: Confidence score representing maximum class probability (float or array)
    """
    # 1. Load model and metadata
    model, metadata = get_baseline_model(model_dir)
    
    # 2. Input validation and shape correction
    if not isinstance(features, np.ndarray):
        raise TypeError(f"Input features must be a numpy ndarray, got {type(features)}")
        
    # Check for NaNs/Infs
    if np.isnan(features).any() or np.isinf(features).any():
        raise ValueError("Input features contain NaNs or infinite values, which are not supported by the model.")

    is_batch = (features.ndim == 2)
    
    if features.ndim == 1:
        if len(features) != NUM_FEATURES:
            raise ValueError(f"Expected exactly {NUM_FEATURES} features, got {len(features)}")
        # Convert to batch of 1
        X = np.expand_dims(features, axis=0)
    elif features.ndim == 2:
        if features.shape[1] != NUM_FEATURES:
            raise ValueError(f"Expected feature dimension to be {NUM_FEATURES}, got {features.shape[1]}")
        X = features
    else:
        raise ValueError(f"Features must have dimension 1 or 2, got shape: {features.shape}")

    # 3. Model Inference
    probs = model.predict_proba(X) # Shape [N, 11]
    class_indices = np.argmax(probs, axis=1) # Shape [N]
    confidences = np.max(probs, axis=1) # Shape [N]
    class_names = np.array([INDEX_TO_MODULATION[idx] for idx in class_indices], dtype=object)

    # 4. Formulate response based on input shape
    if is_batch:
        return {
            "class_index": class_indices.astype(np.int32),
            "class_name": class_names,
            "probabilities": probs.astype(np.float32),
            "confidence": confidences.astype(np.float32)
        }
    else:
        # Squeeze outputs for single sample
        return {
            "class_index": int(class_indices[0]),
            "class_name": str(class_names[0]),
            "probabilities": probs[0].astype(np.float32),
            "confidence": float(confidences[0])
        }

def clear_model_cache():
    """
    Clears the loaded in-memory model cache.
    """
    _MODEL_CACHE.clear()
