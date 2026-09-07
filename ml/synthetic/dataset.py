import os
import json
import numpy as np
from typing import Tuple, List, Dict, Any

from ml.dataset.labels import INDEX_TO_MODULATION, MODULATION_CLASSES

def load_synthetic_dataset(
    data_dir: str = "datasets/synthetic"
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Loads the generated synthetic evaluation dataset and metadata.
    
    Args:
        data_dir (str): Directory containing the synthetic dataset files.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, List[Dict]]:
            - X: Shape [N, 2, 128] float32 array
            - y: Shape [N] int32 array (class index)
            - metadata: List of dict entries matching each sample
    """
    npz_path = os.path.join(data_dir, "synthetic_evaluation_dataset.npz")
    json_path = os.path.join(data_dir, "synthetic_evaluation_metadata.json")
    
    if not os.path.exists(npz_path) or not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Synthetic dataset files not found in '{data_dir}'. "
            f"Please run the generator `ml/synthetic/generator.py` first."
        )
        
    data = np.load(npz_path)
    X = data["X"]
    y = data["y"]
    
    with open(json_path, "r") as f:
        metadata = json.load(f)
        
    # Perform strict validation immediately upon loading
    validate_synthetic_dataset(X, y, metadata)
    
    return X, y, metadata

def validate_synthetic_dataset(
    X: np.ndarray, 
    y: np.ndarray, 
    metadata: List[Dict[str, Any]]
) -> None:
    """
    Validates synthetic dataset array shapes, types, ranges, and metadata alignment.
    
    Args:
        X (np.ndarray): Samples array.
        y (np.ndarray): Labels array.
        metadata (List[Dict]): Metadata list.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError(f"X must be a numpy ndarray, got {type(X)}")
    if not isinstance(y, np.ndarray):
        raise TypeError(f"y must be a numpy ndarray, got {type(y)}")
    if not isinstance(metadata, list):
        raise TypeError(f"metadata must be a list, got {type(metadata)}")
        
    N = X.shape[0]
    
    if X.ndim != 3 or X.shape[1] != 2 or X.shape[2] != 128:
        raise ValueError(f"X shape must be [N, 2, 128], got {X.shape}")
        
    if y.ndim != 1 or y.shape[0] != N:
        raise ValueError(f"y shape must be [N] matching X size {N}, got {y.shape}")
        
    if len(metadata) != N:
        raise ValueError(f"metadata list length ({len(metadata)}) must equal dataset size ({N})")
        
    if X.dtype != np.float32:
        raise TypeError(f"X dtype must be float32, got {X.dtype}")
        
    if not np.issubdtype(y.dtype, np.integer):
        raise TypeError(f"y dtype must be integer, got {y.dtype}")
        
    # Check for NaNs and Infs
    if np.isnan(X).any():
        raise ValueError("Dataset contains NaN values.")
    if np.isinf(X).any():
        raise ValueError("Dataset contains infinite values.")
        
    # Validate each metadata entry
    for i, entry in enumerate(metadata):
        if not isinstance(entry, dict):
            raise TypeError(f"Metadata entry {i} must be a dictionary, got {type(entry)}")
            
        required_keys = ["index", "experiment", "sweep_parameter", "sweep_value", "modulation", "class_index", "random_seed"]
        for key in required_keys:
            if key not in entry:
                raise KeyError(f"Missing required key '{key}' in metadata entry {i}")
                
        # Validate values compatibility
        idx = entry["index"]
        if idx != i:
            raise ValueError(f"Index mismatch in entry {i}: entry['index']={idx}")
            
        mod = entry["modulation"]
        if mod not in MODULATION_CLASSES:
            raise ValueError(f"Unsupported modulation '{mod}' in entry {i}")
            
        c_idx = entry["class_index"]
        if y[i] != c_idx:
            raise ValueError(f"Label alignment mismatch at index {i}: y={y[i]}, metadata['class_index']={c_idx}")
            
        if INDEX_TO_MODULATION[c_idx] != mod:
            raise ValueError(f"Label mapping inconsistency at index {i}: index={c_idx} maps to {INDEX_TO_MODULATION[c_idx]} != {mod}")
            
        snr = entry.get("snr")
        if snr is not None:
            if not isinstance(snr, (int, float, np.number)):
                raise TypeError(f"snr must be numeric, got {type(snr)} at index {i}")
                
        seed = entry["random_seed"]
        if not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed must be integer, got {type(seed)} at index {i}")
