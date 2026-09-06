import os
import torch
import numpy as np
from typing import Dict, Any, Tuple, Union

from ml.dataset.labels import INDEX_TO_MODULATION, MODULATION_CLASSES
from ml.cnn_model.architecture import RawIQCNN

# In-memory model cache to avoid repeated file loading
_MODEL_CACHE: Dict[str, Any] = {}

def get_cnn_model(model_path: str = "models/m5_iq_cnn.pt") -> Tuple[RawIQCNN, float]:
    """
    Loads and caches the PyTorch RawIQCNN baseline model and its training RMS scaling factor.
    
    Args:
        model_path (str): Path to the serialized model state file.
        
    Returns:
        Tuple[RawIQCNN, float]: The PyTorch model (in eval mode) and the RMS scaling factor.
    """
    if "model" in _MODEL_CACHE:
        return _MODEL_CACHE["model"], _MODEL_CACHE["rms_factor"]
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained CNN model checkpoint not found at '{model_path}'. "
            f"Please run the training pipeline script `ml/cnn_model/train.py` first."
        )
        
    print(f"Loading cached PyTorch CNN model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    
    model = RawIQCNN(num_classes=11)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    rms_factor = float(checkpoint["rms_factor"])
    
    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["rms_factor"] = rms_factor
    return model, rms_factor

def predict_iq(iq: np.ndarray, model_path: str = "models/m5_iq_cnn.pt") -> Dict[str, Any]:
    """
    Executes prediction using the trained PyTorch CNN baseline model.
    Supports both single sample (shape [2, 128]) and batch inputs (shape [N, 2, 128]).
    
    Args:
        iq (np.ndarray): NumPy array containing raw float32 IQ samples.
        model_path (str): Path to the serialized model state file.
        
    Returns:
        Dict[str, Any]: Dictionary containing:
            - class_index: Predicted integer class index (int or array)
            - class_name: Predicted canonical modulation name (str or array)
            - probabilities: Predicted class probability vector (array of shape [11] or [N, 11])
            - confidence: Confidence score representing maximum class probability (float or array)
    """
    # 1. Load model and scaling factors
    model, rms_factor = get_cnn_model(model_path)
    
    # 2. Input validation and shape correction
    if not isinstance(iq, np.ndarray):
        raise TypeError(f"Input features must be a numpy ndarray, got {type(iq)}")
        
    # Check for NaNs/Infs
    if np.isnan(iq).any() or np.isinf(iq).any():
        raise ValueError("Input features contain NaNs or infinite values, which are not supported by the model.")

    is_batch = (iq.ndim == 3)
    
    if iq.ndim == 2:
        if iq.shape != (2, 128):
            raise ValueError(f"Expected shape (2, 128) for single IQ sample, got {iq.shape}")
        # Convert to batch of 1
        X = np.expand_dims(iq, axis=0)
    elif iq.ndim == 3:
        if iq.shape[1] != 2 or iq.shape[2] != 128:
            raise ValueError(f"Expected shape [N, 2, 128] for batch, got {iq.shape}")
        X = iq
    else:
        raise ValueError(f"Input IQ must have dimension 2 or 3, got shape: {iq.shape}")

    # 3. Apply normalization scaling
    X_normalized = X / rms_factor
    
    # 4. Model Inference on CPU
    x_tensor = torch.tensor(X_normalized, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(x_tensor)
        probs = torch.softmax(outputs, dim=1).numpy() # Shape [N, 11]
        
    class_indices = np.argmax(probs, axis=1) # Shape [N]
    confidences = np.max(probs, axis=1) # Shape [N]
    class_names = np.array([INDEX_TO_MODULATION[idx] for idx in class_indices], dtype=object)

    # 5. Formulate response based on input shape
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

def clear_cnn_cache():
    """
    Clears the loaded in-memory CNN model cache.
    """
    _MODEL_CACHE.clear()
