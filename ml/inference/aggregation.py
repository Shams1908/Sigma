import numpy as np
from typing import Dict, Tuple
from ml.dataset.labels import INDEX_TO_MODULATION

def aggregate_window_probabilities(
    probs: np.ndarray
) -> Tuple[int, str, float, np.ndarray, Dict[str, int], float]:
    """
    Aggregates window-level prediction probabilities into a single file-level result.
    
    Method:
      1. Calculates the mean probability vector across all analyzed windows.
      2. Appends argmax to obtain the file-level predicted class index.
      3. Extracts confidence, prediction distribution count, and average Shannon entropy.
      
    Args:
        probs (np.ndarray): NumPy array of shape [M, 11] representing softmax probabilities.
        
    Returns:
        Tuple containing:
            - file_class_index (int)
            - file_class_name (str)
            - file_confidence (float)
            - mean_probs (np.ndarray): Mean probability vector of shape [11]
            - prediction_distribution (Dict[str, int]): Count of windows predicted as each class
            - mean_entropy (float): Mean Shannon entropy of predictions across windows
    """
    if len(probs) == 0:
        raise ValueError("Cannot aggregate probabilities for an empty segment array.")
        
    # Calculate the mean probability vector across all windows (axis=0)
    mean_probs = np.mean(probs, axis=0)
    
    # Choose class with the highest average probability
    file_class_index = int(np.argmax(mean_probs))
    file_class_name = INDEX_TO_MODULATION[file_class_index]
    file_confidence = float(mean_probs[file_class_index])
    
    # Calculate window-level prediction distribution counts
    window_preds = np.argmax(probs, axis=1)
    prediction_distribution = {}
    unique_idxs, counts = np.unique(window_preds, return_counts=True)
    for idx, count in zip(unique_idxs, counts):
        cls_name = INDEX_TO_MODULATION[int(idx)]
        prediction_distribution[cls_name] = int(count)
        
    # Calculate Shannon Entropy per window: H = -sum(p * log(p))
    window_entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    mean_entropy = float(np.mean(window_entropy))
    
    return file_class_index, file_class_name, file_confidence, mean_probs, prediction_distribution, mean_entropy
