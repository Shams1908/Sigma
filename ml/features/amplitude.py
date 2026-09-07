import numpy as np

def extract_amplitude_features(x: np.ndarray) -> dict:
    """
    Extract amplitude-based features from a complex signal array.
    Supports both 1D arrays [128,] and 2D batch arrays [N, 128].
    
    Args:
        x (np.ndarray): Complex signal array (complex64 or complex128).
        
    Returns:
        dict: Dictionary containing extracted amplitude features.
    """
    # Force conversion to double precision for numerical stability in statistics
    x_complex = x.astype(np.complex128)
    a = np.abs(x_complex) # Amplitude
    
    # Calculate statistics along the time axis (last dimension)
    mean_a = np.mean(a, axis=-1)
    var_a = np.var(a, axis=-1)
    
    # Excess Kurtosis calculation: E[(a - E[a])^4] / Var(a)^2 - 3
    # Use broadcasting for mean subtraction
    # If 1D: a - mean_a (scalar broadcasted)
    # If 2D: a - mean_a[:, None] (need explicit None/expand_dims for proper broadcasting)
    if x.ndim == 2:
        a_diff = a - np.expand_dims(mean_a, axis=-1)
    else:
        a_diff = a - mean_a
        
    m4 = np.mean(a_diff**4, axis=-1)
    
    # Stable division to avoid NaNs/Infs for zero/near-zero variance
    # If var_a is near-zero, excess kurtosis is set to 0.0
    kurtosis_a = np.where(var_a > 1e-9, m4 / np.maximum(var_a**2, 1e-12) - 3.0, 0.0)
    
    # Peak-to-Average Power Ratio (PAR): max(a)^2 / mean(a^2)
    max_a = np.max(a, axis=-1)
    mean_a2 = np.mean(a**2, axis=-1)
    par_a = np.where(mean_a2 > 1e-9, (max_a**2) / np.maximum(mean_a2, 1e-12), 0.0)
    
    # Cast outputs to float32 for final representation
    return {
        "amplitude_mean": mean_a.astype(np.float32),
        "amplitude_variance": var_a.astype(np.float32),
        "amplitude_kurtosis": kurtosis_a.astype(np.float32),
        "amplitude_peak_to_average_ratio": par_a.astype(np.float32)
    }
