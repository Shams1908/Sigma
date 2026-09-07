import numpy as np

def extract_frequency_features(x: np.ndarray) -> dict:
    """
    Extract frequency-based features from a complex signal array using the phase increment representation.
    Supports both 1D arrays [128,] and 2D batch arrays [N, 128].
    
    Note: These are normalized per-sample quantities (angular frequency change in radians per sample)
    and are not converted into physical Hz, as the physical sample rate is not provided.
    
    Args:
        x (np.ndarray): Complex signal array.
        
    Returns:
        dict: Dictionary containing extracted frequency features.
    """
    # Force conversion to double precision
    x_complex = x.astype(np.complex128)
    
    # Calculate phase increments dphi[n] = angle(x[n] * conj(x[n-1]))
    x_curr = x_complex[..., 1:]
    x_lag = x_complex[..., :-1]
    prod = x_curr * np.conj(x_lag)
    prod_mag = np.abs(prod)
    
    dphi = np.angle(prod)
    # Safely handle zero or near-zero signal power
    dphi = np.where(prod_mag > 1e-18, dphi, 0.0)
    
    # Instantaneous frequency statistics
    mean_if = np.mean(dphi, axis=-1)
    var_if = np.var(dphi, axis=-1)
    
    return {
        "instantaneous_frequency_mean": mean_if.astype(np.float32),
        "instantaneous_frequency_variance": var_if.astype(np.float32)
    }
