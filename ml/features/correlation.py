import numpy as np

def extract_autocorrelation_features(x: np.ndarray) -> dict:
    """
    Extract normalized complex autocorrelation features for lags [1, 2, 4, 8, 16].
    Supports both 1D arrays [128,] and 2D batch arrays [N, 128].
    
    Formula:
        R[k] = E[x[n] * conj(x[n-k])]
        R_norm[k] = R[k] / R[0]
        
    Args:
        x (np.ndarray): Complex signal array.
        
    Returns:
        dict: Dictionary containing real, imaginary, and magnitude features for each lag.
    """
    # Force conversion to double precision for numerical stability
    x_complex = x.astype(np.complex128)
    
    # R[0] is the average power of the signal
    R0 = np.mean(np.abs(x_complex)**2, axis=-1)
    
    lags = [1, 2, 4, 8, 16]
    features = {}
    
    for k in lags:
        # Get slices for lag calculation
        x_curr = x_complex[..., k:]
        x_lag = x_complex[..., :-k]
        
        # Calculate sample mean over overlapping portion
        R_k = np.mean(x_curr * np.conj(x_lag), axis=-1)
        
        # Normalize by R0 and handle zero-power signal safely
        R_norm_k = np.where(R0 > 1e-9, R_k / np.maximum(R0, 1e-12), 0.0 + 0.0j)
        
        features[f"autocorr_lag_{k}_real"] = np.real(R_norm_k).astype(np.float32)
        features[f"autocorr_lag_{k}_imag"] = np.imag(R_norm_k).astype(np.float32)
        features[f"autocorr_lag_{k}_magnitude"] = np.abs(R_norm_k).astype(np.float32)
        
    return features
