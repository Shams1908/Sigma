from typing import Tuple, List, Optional
import numpy as np
from ml.dataset.loader import RadioMLDataset
from ml.generators.signal import iq_to_complex

from ml.features.schema import FEATURE_NAMES, NUM_FEATURES
from ml.features.amplitude import extract_amplitude_features
from ml.features.phase import extract_phase_features
from ml.features.frequency import extract_frequency_features
from ml.features.cumulants import extract_cumulant_features
from ml.features.correlation import extract_autocorrelation_features

def get_feature_names() -> List[str]:
    """
    Returns the central, stable ordered list of feature names.
    """
    return list(FEATURE_NAMES)

def extract_features(samples: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """
    Extract features from a single signal example.
    
    Args:
        samples (np.ndarray): IQ samples of shape [2, 128].
        
    Returns:
        Tuple[List[str], np.ndarray]: 
            - list of 36 feature names in deterministic order
            - 1D float32 NumPy array of shape [36]
    """
    if samples.shape != (2, 128):
        raise ValueError(f"Expected single sample of shape (2, 128), got {samples.shape}")
        
    # Convert IQ to 1D complex array of shape (128,)
    x = iq_to_complex(samples)
    
    # Extract features from each family
    feature_dict = {}
    feature_dict.update(extract_amplitude_features(x))
    feature_dict.update(extract_phase_features(x))
    feature_dict.update(extract_frequency_features(x))
    feature_dict.update(extract_cumulant_features(x))
    feature_dict.update(extract_autocorrelation_features(x))
    
    # Construct 1D array in the exact schema order
    feature_vector = np.array([feature_dict[name] for name in FEATURE_NAMES], dtype=np.float32)
    return get_feature_names(), feature_vector

def extract_batch_features(batch_samples: np.ndarray) -> np.ndarray:
    """
    Vectorized feature extraction for a batch of signals.
    
    Args:
        batch_samples (np.ndarray): IQ samples of shape [N, 2, 128].
        
    Returns:
        np.ndarray: Float32 feature matrix of shape [N, 36].
    """
    if batch_samples.ndim != 3 or batch_samples.shape[1] != 2:
        raise ValueError(f"Expected batch samples of shape [N, 2, L], got {batch_samples.shape}")
        
    N = batch_samples.shape[0]
    
    # Reconstruct complex numbers in a vectorized way: shape [N, L]
    x = batch_samples[:, 0].astype(np.complex64) + 1j * batch_samples[:, 1].astype(np.complex64)
    
    # Extract features from each family (vectorized across the batch)
    feature_dict = {}
    feature_dict.update(extract_amplitude_features(x))
    feature_dict.update(extract_phase_features(x))
    feature_dict.update(extract_frequency_features(x))
    feature_dict.update(extract_cumulant_features(x))
    feature_dict.update(extract_autocorrelation_features(x))
    
    # Stack features into a 2D matrix of shape [N, 36]
    feature_matrix = np.column_stack([feature_dict[name] for name in FEATURE_NAMES]).astype(np.float32)
    return feature_matrix

def extract_feature_matrix(
    dataset: RadioMLDataset, 
    indices: Optional[np.ndarray] = None,
    chunk_size: int = 50000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the feature matrix, labels, and SNR metadata for a given dataset subset.
    Processes in configurable chunks for memory safety.
    
    Args:
        dataset (RadioMLDataset): The dataset loader containing samples.
        indices (np.ndarray, optional): Subset indices. If None, uses all dataset samples.
        chunk_size (int): Size of chunks to process at a time.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - X: Feature matrix of shape [N, 36]
            - y: Target modulation class indices of shape [N]
            - snrs: SNR values of shape [N]
    """
    if indices is None:
        indices = np.arange(len(dataset), dtype=np.int32)
        
    total_samples = len(indices)
    
    # Allocate the full feature matrix
    X = np.empty((total_samples, NUM_FEATURES), dtype=np.float32)
    y = np.empty(total_samples, dtype=np.int32)
    snrs = np.empty(total_samples, dtype=np.int32)
    
    # Retrieve flat arrays from the dataset cache using indices
    # We do index slicing here to avoid copying the full raw dataset
    all_raw_samples = dataset.samples # Shape [total_samples, 2, 128]
    all_class_indices = dataset.class_indices
    all_snrs = dataset.snrs
    
    # Process in chunks
    for i in range(0, total_samples, chunk_size):
        end_idx = min(i + chunk_size, total_samples)
        chunk_samples = all_raw_samples[i:end_idx]
        
        # Extract features for this chunk
        X[i:end_idx] = extract_batch_features(chunk_samples)
        y[i:end_idx] = all_class_indices[i:end_idx]
        snrs[i:end_idx] = all_snrs[i:end_idx]
        
    return X, y, snrs
